'use strict';

const compilerName = /^(?:macOS arm64 Metal|Ubuntu (?:x64|arm64) CPU|Ubuntu x64 (?:Vulkan|ROCm|SYCL|CUDA [0-9]+(?:\.[0-9]+)*)|Windows x64 (?:CPU|SYCL|Vulkan|HIP\/Radeon|CUDA [0-9]+(?:\.[0-9]+)*))$/;
const incompletePrefixes = [
    'BEELLAMA_BUILD_INCOMPLETE:',
    'The hosted runner lost communication with the server.',
];

function validate(inputs, runId) {
    const safeVersion = /^v[0-9][0-9A-Za-z._-]*$/;
    if (!inputs || typeof inputs.source_ref !== 'string' || !safeVersion.test(inputs.source_ref) ||
        typeof inputs.source_sha !== 'string' || !/^[0-9a-f]{40}$/.test(inputs.source_sha)) {
        throw new Error('Invalid release source inputs.');
    }
    let publish;
    if (inputs.publish_release === true || inputs.publish_release === 'true') publish = true;
    else if (inputs.publish_release === false || inputs.publish_release === 'false') publish = false;
    else throw new Error('publish_release must be a boolean.');
    if (publish ? inputs.tag_name !== inputs.source_ref : inputs.tag_name !== `preview-${inputs.source_ref}`) {
        throw new Error('Release tag does not match the source and publication mode.');
    }
    const attempt = inputs.retry_attempt ?? '0';
    if (typeof attempt !== 'string' || !/^[012]$/.test(attempt)) {
        throw new Error('retry_attempt must be 0, 1, or 2.');
    }
    const suppliedRoot = inputs.retry_root ?? '';
    if (typeof suppliedRoot !== 'string' || (suppliedRoot !== '' && !/^[1-9][0-9]*$/.test(suppliedRoot)) ||
        (attempt !== '0' && suppliedRoot === '')) {
        throw new Error('Retries require a numeric retry_root.');
    }
    const root = suppliedRoot || String(runId);
    if (!/^[1-9][0-9]*$/.test(root)) throw new Error('Invalid workflow run ID.');
    return { publish, attempt: Number(attempt), root };
}

// Explicit pagination keeps every request bounded and lets API failures fail the gate closed.
async function pages(method, args, collection) {
    const items = [];
    for (let page = 1; ; page++) {
        const { data } = await method({ ...args, per_page: 100, page });
        const batch = collection ? data[collection] : data;
        items.push(...batch);
        if (batch.length < 100) return items;
    }
}

module.exports = async function retryRelease({ github, context, core, inputs, results }) {
    core.setOutput('proceed', 'false');
    const { publish, attempt, root } = validate(inputs, context.runId);
    if (!results || typeof results !== 'object' || Array.isArray(results) || Object.keys(results).length === 0 ||
        Object.values(results).some(value => !value || typeof value.result !== 'string')) {
        throw new Error('Dependency results must be a nonempty needs mapping.');
    }
    const { owner, repo } = context.repo;
    const repoArgs = { owner, repo };
    const runArgs = { ...repoArgs, run_id: context.runId };
    const getCurrent = async () => (await github.rest.actions.getWorkflowRun(runArgs)).data;
    const cancelled = run => run.conclusion === 'cancelled' || run.status === 'cancelled';
    const current = await getCurrent();
    if (cancelled(current)) return;
    if (!current.created_at || !Number.isFinite(Date.parse(current.created_at))) {
        throw new Error('Current run has no valid creation time for the retry search.');
    }
    // A child can have finished before a manual rerun. Do not filter by active status.
    const runs = await pages(github.rest.actions.listWorkflowRuns, {
        ...repoArgs, workflow_id: 'release.yml', event: 'workflow_dispatch', branch: 'main',
        created: `>=${current.created_at}`,
    }, 'workflow_runs');
    const mainDispatches = runs.filter(run => run.head_branch === 'main' && run.event === 'workflow_dispatch');
    const baseTitle = `Build ${inputs.source_ref} Preview`;
    const superseded = !publish && mainDispatches.some(run => Number(run.id) > Number(context.runId) &&
        (run.display_title === baseTitle || run.display_title?.startsWith(`${baseTitle} [chain `)));
    if (superseded) return;

    const sourceCurrent = async () => {
        if (publish) return true;
        const { data } = await github.rest.repos.getBranch({ ...repoArgs, branch: inputs.source_ref });
        return data.commit.sha === inputs.source_sha;
    };
    if (!await sourceCurrent()) return;
    if (Object.values(results).every(value => value.result === 'success')) {
        if (!cancelled(await getCurrent())) core.setOutput('proceed', 'true');
        return;
    }
    if (attempt >= 2) return;
    const hasChild = mainDispatches.some(run => {
        const suffix = / \[chain ([1-9][0-9]*) attempt ([012])\]$/.exec(run.display_title ?? '');
        return suffix && suffix[1] === root && Number(suffix[2]) > attempt;
    });
    if (hasChild) return;

    const jobs = await pages(github.rest.actions.listJobsForWorkflowRun, { ...runArgs, filter: 'latest' }, 'jobs');
    let incomplete = false;
    for (const job of jobs.filter(job => compilerName.test(job.name))) {
        if (job.conclusion === 'timed_out') {
            incomplete = true;
            break;
        }
        // Positive worker evidence is useful even when the job itself was cancelled.
        if (!job.check_run_url) continue;
        const id = /\/check-runs\/([0-9]+)$/.exec(job.check_run_url);
        if (!id) throw new Error('Compiler job has an invalid check-run URL.');
        const annotations = await pages(github.rest.checks.listAnnotations, {
            ...repoArgs, check_run_id: Number(id[1]),
        });
        if (annotations.some(annotation => typeof annotation.message === 'string' &&
            incompletePrefixes.some(prefix => annotation.message.startsWith(prefix)))) {
            incomplete = true;
            break;
        }
    }
    if (!incomplete || cancelled(await getCurrent()) || !await sourceCurrent()) return;
    await github.rest.actions.createWorkflowDispatch({
        ...repoArgs, workflow_id: 'release.yml', ref: 'main',
        inputs: {
            source_ref: inputs.source_ref,
            source_sha: inputs.source_sha,
            tag_name: inputs.tag_name,
            publish_release: String(publish),
            retry_attempt: String(attempt + 1),
            retry_root: root,
        },
    });
};
