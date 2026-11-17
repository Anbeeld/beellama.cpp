'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const retry = require('../.github/scripts/retry-release.cjs');

const sha = 'a'.repeat(40);
function fixture(options = {}) {
    const inputs = { source_ref: 'v0.4.0', source_sha: sha, tag_name: 'preview-v0.4.0',
        publish_release: false, retry_attempt: '0', retry_root: '', ...options.inputs };
    const calls = { dispatches: [], jobs: [], annotations: [], runs: [], outputs: {} };
    const current = { id: 100, status: 'in_progress', conclusion: null,
        created_at: '2026-04-01T00:00:00Z', ...options.current };
    const job = { name: 'Ubuntu x64 CUDA 13.3', conclusion: 'failure',
        check_run_url: 'https://api.github.com/repos/owner/repo/check-runs/42' };
    const github = { rest: {
        actions: {
            getWorkflowRun: async () => ({ data: current }),
            listJobsForWorkflowRun: async args => {
                calls.jobs.push(args);
                return { data: { jobs: options.jobPages?.[args.page - 1] ??
                    (args.page === 1 ? options.jobs ?? [job] : []) } };
            },
            listWorkflowRuns: async args => {
                calls.runs.push(args);
                return { data: { workflow_runs: options.runPages?.[args.page - 1] ??
                    (args.page === 1 ? options.runs ?? [] : []) } };
            },
            createWorkflowDispatch: async args => { calls.dispatches.push(args); },
        },
        checks: { listAnnotations: async args => {
            calls.annotations.push(args);
            return { data: options.annotationPages?.[args.page - 1] ??
                (args.page === 1 ? options.annotations ?? [] : []) };
        } },
        repos: { getBranch: async () => ({ data: { commit: { sha: options.branchSha ?? sha } } }) },
    } };
    const args = { github, context: { repo: { owner: 'owner', repo: 'repo' }, runId: 100 },
        core: { setOutput: (key, value) => { calls.outputs[key] = value; } }, inputs,
        results: options.results ?? { 'release-meta': { result: 'success' }, 'ubuntu-cuda': { result: 'failure' } } };
    return { args, calls, current, run: () => retry(args) };
}
const marker = { message: 'BEELLAMA_BUILD_INCOMPLETE: third checkpoint saved' };

test('healthy dependencies proceed without retry', async () => {
    const f = fixture({ results: { meta: { result: 'success' }, build: { result: 'success' } } });
    await f.run();
    assert.equal(f.calls.outputs.proceed, 'true');
    assert.equal(f.calls.dispatches.length, 0);
});

test('plain compiler failure does not dispatch', async () => {
    const f = fixture(); await f.run();
    assert.equal(f.calls.outputs.proceed, 'false');
    assert.equal(f.calls.dispatches.length, 0);
});

for (const [label, annotation] of [
    ['checkpoint marker', marker],
    ['observed worker loss', { message: 'The hosted runner lost communication with the server. Please try again.' }],
]) test(`${label} dispatches and preserves inputs`, async () => {
    const f = fixture({ annotations: [annotation] }); await f.run();
    assert.deepEqual(f.calls.dispatches, [{ owner: 'owner', repo: 'repo', workflow_id: 'release.yml', ref: 'main',
        inputs: { ...f.args.inputs, publish_release: 'false', retry_attempt: '1', retry_root: '100' } }]);
    assert.equal(f.calls.outputs.proceed, 'false');
});

test('mixed ordinary compile error and incomplete dispatch once', async () => {
    const f = fixture({ annotations: [marker], jobs: [
        { name: 'Ubuntu x64 CPU', conclusion: 'failure' },
        { name: 'Windows x64 HIP/Radeon', conclusion: 'cancelled', check_run_url: 'https://api.github.com/repos/o/r/check-runs/12' },
    ] }); await f.run();
    assert.equal(f.calls.dispatches.length, 1);
    assert.equal(f.calls.annotations[0].check_run_id, 12);
});

test('unknown failures and job cancellations are not incomplete', async () => {
    const f = fixture({ jobs: [{ name: 'Windows x64 CPU', conclusion: 'cancelled',
        check_run_url: 'https://api.github.com/repos/o/r/check-runs/12' }],
        annotations: [{ message: 'Runner shutdown requested' }, { message: 'error: BEELLAMA_BUILD_INCOMPLETE: not a prefix' }] });
    await f.run(); assert.equal(f.calls.dispatches.length, 0);
});

test('hosted compiler timeout dispatches; unrelated jobs ignored', async () => {
    const f = fixture({ jobs: [{ name: 'Ubuntu arm64 CPU', conclusion: 'timed_out' }] });
    await f.run(); assert.equal(f.calls.dispatches.length, 1);
    const unrelated = fixture({ jobs: [{ name: 'Publication metadata', conclusion: 'timed_out' },
        { name: 'retry-builds', conclusion: 'timed_out' }, { name: 'Ubuntu x64 CUDA impostor', conclusion: 'timed_out' }] });
    await unrelated.run(); assert.equal(unrelated.calls.dispatches.length, 0);
});

for (const [label, options] of [
    ['branch moved', { branchSha: 'b'.repeat(40) }],
    ['cancelled run', { current: { conclusion: 'cancelled' } }],
    ['newer same-source preview', { runs: [{ id: 101, head_branch: 'main', event: 'workflow_dispatch',
        display_title: 'Build v0.4.0 Preview [chain 101 attempt 0]' }] }],
]) test(`${label} suppresses retry and proceeding`, async () => {
    for (const healthy of [false, true]) {
        const f = fixture({ ...options, annotations: [marker], ...(healthy ? { results: { build: { result: 'success' } } } : {}) });
        await f.run(); assert.equal(f.calls.outputs.proceed, 'false'); assert.equal(f.calls.dispatches.length, 0);
    }
});

test('unrelated source/mode/branch runs do not supersede preview', async () => {
    const f = fixture({ annotations: [marker], runs: [
        { id: 101, head_branch: 'main', event: 'workflow_dispatch', display_title: 'Build v0.5.0 Preview [chain 101 attempt 0]' },
        { id: 102, head_branch: 'main', event: 'workflow_dispatch', display_title: 'Release v0.4.0 [chain 102 attempt 0]' },
        { id: 103, head_branch: 'other', event: 'workflow_dispatch', display_title: 'Build v0.4.0 Preview [chain 103 attempt 0]' },
    ] }); await f.run(); assert.equal(f.calls.dispatches.length, 1);
});

test('attempt cap is 0 -> 1 -> 2 -> stop', async () => {
    for (let attempt = 0; attempt <= 2; attempt++) {
        const f = fixture({ inputs: { retry_attempt: String(attempt), retry_root: attempt ? '90' : '' }, annotations: [marker] });
        await f.run();
        assert.equal(f.calls.dispatches.length, attempt < 2 ? 1 : 0);
        if (attempt < 2) assert.equal(f.calls.dispatches[0].inputs.retry_attempt, String(attempt + 1));
    }
});

test('stable boolean string dispatches preserved stable metadata', async () => {
    const f = fixture({ inputs: { publish_release: 'true', tag_name: 'v0.4.0' }, annotations: [marker], branchSha: 'b'.repeat(40) });
    await f.run(); assert.equal(f.calls.dispatches[0].inputs.publish_release, 'true');
    assert.equal(f.calls.dispatches[0].inputs.tag_name, 'v0.4.0');
});

test('finished child suppresses duplicate dispatch on manual rerun', async () => {
    const f = fixture({ inputs: { publish_release: true, tag_name: 'v0.4.0' }, annotations: [marker], runs: [
        { id: 101, head_branch: 'main', event: 'workflow_dispatch', status: 'completed',
            display_title: 'Release v0.4.0 [chain 100 attempt 1]' },
    ] }); await f.run(); assert.equal(f.calls.dispatches.length, 0);
});

test('paginates jobs, annotations, and bounded workflow runs', async () => {
    const f = fixture({
        jobPages: [Array.from({ length: 100 }, () => ({ name: 'metadata' })), [
            { name: 'macOS arm64 Metal', conclusion: 'failure', check_run_url: 'https://api.github.com/repos/o/r/check-runs/99' }]],
        annotationPages: [Array.from({ length: 100 }, () => ({ message: 'ordinary' })), [marker]],
        runPages: [Array.from({ length: 100 }, () => ({ id: 1 })), []],
    }); await f.run();
    assert.equal(f.calls.jobs.length, 2); assert.equal(f.calls.annotations.length, 2);
    assert.equal(f.calls.runs.length, 2); assert.equal(f.calls.dispatches.length, 1);
    assert.equal(f.calls.jobs[0].filter, 'latest');
    assert.equal(f.calls.runs[0].created, '>=2026-04-01T00:00:00Z');
});

test('rechecks cancellation immediately before dispatch', async () => {
    const f = fixture({ annotations: [marker] });
    const original = f.args.github.rest.actions.listJobsForWorkflowRun;
    f.args.github.rest.actions.listJobsForWorkflowRun = async args => {
        const result = await original(args); f.current.conclusion = 'cancelled'; return result;
    };
    await f.run(); assert.equal(f.calls.dispatches.length, 0);
});

for (const [group, method] of [['actions', 'getWorkflowRun'], ['actions', 'listWorkflowRuns'],
    ['actions', 'listJobsForWorkflowRun'], ['checks', 'listAnnotations'], ['repos', 'getBranch'],
    ['actions', 'createWorkflowDispatch']]) test(`${method} API failure propagates`, async () => {
    const f = fixture({ annotations: [marker] });
    const error = new Error('API unavailable');
    f.args.github.rest[group][method] = async () => { throw error; };
    await assert.rejects(f.run, error); assert.equal(f.calls.outputs.proceed, 'false');
});

for (const inputs of [
    { source_sha: 'abc' }, { source_sha: 'g'.repeat(40) }, { source_ref: '../v0.4.0' },
    { source_ref: ['v0.4.0'] },
    { tag_name: 'preview-other' }, { publish_release: 'yes' }, { publish_release: true },
    { retry_attempt: '-1' }, { retry_attempt: '3' }, { retry_attempt: '1.0' },
    { retry_attempt: '1', retry_root: '' }, { retry_root: 'unsafe' },
]) test(`rejects malformed inputs ${JSON.stringify(inputs)}`, async () => {
    const f = fixture({ inputs }); await assert.rejects(f.run);
    assert.equal(f.calls.dispatches.length, 0); assert.equal(f.calls.outputs.proceed, 'false');
});
