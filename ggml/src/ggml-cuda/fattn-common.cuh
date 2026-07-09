#pragma once

#include "common.cuh"
#include "convert.cuh"
#include "vecdotq.cuh"
#include "turbo4-tcq-codebook.cuh"

#include <cstdint>

static __constant__ float d_turbo_centroids_2bit_fattn[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f
};
static __constant__ float d_turbo_centroids_3bit_fattn[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};
static __constant__ float d_turbo_centroids_4bit_fattn[16] = {
    -0.241556f, -0.182907f, -0.143047f, -0.111065f,
    -0.083317f, -0.058069f, -0.034311f, -0.011353f,
     0.011353f,  0.034311f,  0.058069f,  0.083317f,
     0.111065f,  0.143047f,  0.182907f,  0.241556f,
};

// 3-bit TCQ codebook (product_mono/iter080, 512-state bitshift trellis). If you copy these, credit spiritbuun!
// CUDA GLA product-aware training, 100 iters on Qwen3.5-27B FWHT-rotated KV activations. Decode: state_t = read_9_bits(qs, t*3)
static __constant__ float d_turbo3_tcq_codebook_fattn[512] = {
    -0.14559399f, -0.09062801f, -0.054925077f, -0.03699251f, -0.006363985f, +0.026264573f, +0.067378916f, +0.121981815f,
    -0.18648055f, -0.106522456f, -0.052047577f, -0.011695214f, +0.021953275f, +0.059698727f, +0.09831437f, +0.16083933f,
    -0.16390342f, -0.12639847f, -0.09513180f, -0.05938352f, -0.028396897f, +0.005973862f, +0.049104784f, +0.11334257f,
    -0.25952467f, -0.079778515f, -0.036024813f, +0.0003641268f, +0.031858794f, +0.073280424f, +0.11835553f, +0.19738495f,
    -0.14218009f, -0.10224814f, -0.062498566f, -0.027066832f, +0.00393002f, +0.04069300f, +0.08257346f, +0.14548601f,
    -0.18673635f, -0.13438253f, -0.088401966f, -0.05205436f, -0.02032501f, +0.012399545f, +0.05127183f, +0.10316186f,
    -0.10807011f, -0.065903045f, -0.032206114f, -0.0062006037f, +0.020679146f, +0.04422085f, +0.08313074f, +0.16821936f,
    -0.22979105f, -0.14431947f, -0.07689272f, -0.02755307f, +0.009225173f, +0.046684854f, +0.08834142f, +0.13766693f,
    -0.22114082f, -0.12612148f, -0.06890522f, -0.016128855f, +0.03691900f, +0.08474852f, +0.14940020f, +0.23229980f,
    -0.14933491f, -0.099693604f, -0.06738499f, -0.037100967f, -0.009332986f, +0.023535024f, +0.060272533f, +0.109464675f,
    -0.20200425f, -0.07398328f, -0.038700905f, -0.01714807f, +0.011161969f, +0.04528101f, +0.08902637f, +0.19573534f,
    -0.16645233f, -0.124482535f, -0.089342155f, -0.04427387f, -0.007353691f, +0.028033108f, +0.066108435f, +0.15552913f,
    -0.22295763f, -0.059887577f, -0.018804537f, +0.020141022f, +0.059682943f, +0.097920544f, +0.14080113f, +0.25698325f,
    -0.14248224f, -0.089685425f, -0.050101686f, -0.017257255f, +0.011412255f, +0.040830314f, +0.07400172f, +0.11997315f,
    -0.18649384f, -0.113997504f, -0.067775466f, -0.033394672f, +0.006586988f, +0.05312057f, +0.10433043f, +0.22344802f,
    -0.16138338f, -0.108194515f, -0.07600300f, -0.05135381f, -0.023365447f, +0.0087320795f, +0.045431953f, +0.09113002f,
    -0.12630440f, -0.07225349f, -0.032280035f, +0.0029231994f, +0.019239848f, +0.05081419f, +0.077840395f, +0.121695265f,
    -0.08928155f, -0.044983763f, -0.009889568f, +0.020831043f, +0.05684458f, +0.09409702f, +0.13867535f, +0.19084482f,
    -0.14182915f, -0.11380146f, -0.06904074f, -0.002002765f, +0.034864165f, +0.070399575f, +0.11403063f, +0.15394832f,
    -0.10876417f, -0.056122433f, -0.02267638f, +0.011113975f, +0.039639056f, +0.074084364f, +0.10155376f, +0.12540291f,
    -0.17693359f, -0.13940524f, -0.10049578f, -0.06796275f, -0.036915872f, +0.00062823476f, +0.042142134f, +0.17906062f,
    -0.09253492f, -0.04290128f, -0.006311852f, +0.023908244f, +0.049849935f, +0.078770354f, +0.10818172f, +0.15166481f,
    -0.12429565f, -0.07392063f, -0.029114135f, +0.0059440783f, +0.042675965f, +0.08425635f, +0.13836108f, +0.18634140f,
    -0.11795639f, -0.07033707f, -0.034163877f, -0.0008773357f, +0.03334606f, +0.07188203f, +0.12216825f, +0.17097956f,
    -0.18718453f, -0.14090346f, -0.097799584f, -0.059522875f, -0.019208657f, +0.03079176f, +0.09334672f, +0.15811224f,
    -0.27198875f, -0.16546582f, -0.11433405f, -0.06933013f, -0.04026183f, -0.0061146915f, +0.029263576f, +0.07322499f,
    -0.18471734f, -0.102074504f, -0.06492570f, -0.034418534f, -0.009636157f, +0.023043344f, +0.05751496f, +0.09905984f,
    -0.22826399f, -0.15946552f, -0.09913176f, -0.06585259f, -0.03252090f, +0.001313243f, +0.03556729f, +0.21612854f,
    -0.13243781f, -0.087299444f, -0.049820945f, -0.016216082f, +0.01799807f, +0.057916876f, +0.09001349f, +0.13221787f,
    -0.19516511f, -0.120894566f, -0.076130204f, -0.051442243f, -0.029535033f, -0.0020043184f, +0.029452588f, +0.075566076f,
    -0.27272871f, -0.15841717f, -0.105432935f, -0.06792948f, -0.024532158f, +0.014960791f, +0.054415092f, +0.101517834f,
    -0.21153601f, -0.15015371f, -0.08676790f, -0.04414934f, -0.0042129597f, +0.033762872f, +0.07589151f, +0.12768789f,
    -0.090428725f, -0.037582967f, +0.0013173596f, +0.03900247f, +0.06840049f, +0.116906695f, +0.16584939f, +0.25382105f,
    -0.13446195f, -0.07865091f, -0.039625354f, -0.0028398742f, +0.03019514f, +0.06799379f, +0.11850997f, +0.17521496f,
    -0.11350345f, -0.058599845f, -0.017512511f, +0.019431496f, +0.055897832f, +0.093173414f, +0.14820710f, +0.22092152f,
    -0.15165758f, -0.08869354f, -0.04974287f, -0.01705474f, +0.013134752f, +0.04367713f, +0.07733791f, +0.12430801f,
    -0.09329869f, -0.04673005f, -0.00045857552f, +0.042781368f, +0.07802363f, +0.11887439f, +0.16250038f, +0.28612965f,
    -0.12571070f, -0.07786012f, -0.03843933f, -0.0075433915f, +0.025822964f, +0.066053316f, +0.12021536f, +0.18341768f,
    -0.16079275f, -0.04921760f, -0.006114644f, +0.026215268f, +0.05699377f, +0.09813471f, +0.16080129f, +0.23786584f,
    -0.09980837f, -0.048535258f, -0.0096120685f, +0.025387142f, +0.05979822f, +0.09875251f, +0.14474337f, +0.20324114f,
    -0.15846540f, -0.09938028f, -0.061492465f, -0.03523542f, -0.0061364113f, +0.024916094f, +0.06037314f, +0.106796466f,
    -0.20557843f, -0.123237535f, -0.07734871f, -0.044549115f, -0.017114898f, +0.01616654f, +0.049574375f, +0.092319444f,
    -0.19221115f, -0.14642999f, -0.091701314f, -0.055265956f, -0.021026207f, +0.017720066f, +0.05786183f, +0.110154524f,
    -0.09956386f, -0.03870283f, +0.003052007f, +0.034851722f, +0.06256365f, +0.09628840f, +0.13979156f, +0.16582295f,
    -0.18026546f, -0.12448310f, -0.07424377f, -0.03954519f, -0.01221123f, +0.028641058f, +0.100819774f, +0.18240699f,
    -0.21520759f, -0.15573645f, -0.09820838f, -0.051450998f, -0.012993679f, +0.021135861f, +0.058727216f, +0.105848536f,
    -0.11207385f, -0.08335689f, -0.048542723f, -0.023198519f, +0.0039304253f, +0.037778318f, +0.07813917f, +0.13106476f,
    -0.17849164f, -0.120988995f, -0.078016765f, -0.043093704f, -0.016565649f, +0.015182641f, +0.050754096f, +0.09595712f,
    -0.22132620f, -0.13407415f, -0.065785654f, -0.013291034f, +0.032098345f, +0.07478225f, +0.12431934f, +0.19174045f,
    -0.095454164f, -0.051898945f, -0.015116375f, -0.012596778f, +0.018636847f, +0.05006925f, +0.087654814f, +0.13754296f,
    -0.15254061f, -0.09576059f, -0.052086458f, -0.01596074f, +0.017607626f, +0.04778498f, +0.08950204f, +0.14901252f,
    -0.26057002f, -0.12472382f, -0.074396215f, -0.03764066f, +0.0011168446f, +0.061569117f, +0.10793752f, +0.19771695f,
    -0.08661132f, -0.045195263f, -0.016098704f, +0.012780116f, +0.040476497f, +0.074102715f, +0.074102715f, +0.12635531f,
    -0.14047913f, -0.059587404f, -0.016261123f, +0.019801628f, +0.053541403f, +0.096650146f, +0.15005490f, +0.21051759f,
    -0.22986396f, -0.11964334f, -0.07266585f, -0.026522418f, +0.018169926f, +0.058630653f, +0.100647695f, +0.15919648f,
    -0.13251697f, -0.077567816f, -0.042766172f, -0.011389967f, +0.01831755f, +0.05304656f, +0.09620367f, +0.15567583f,
    -0.119819686f, -0.06772876f, -0.028123451f, +0.00876240f, +0.014405836f, +0.048829112f, +0.08422175f, +0.13823749f,
    -0.16379014f, -0.08956941f, -0.041652776f, +0.008921398f, +0.05473602f, +0.10037984f, +0.16022855f, +0.23457925f,
    -0.115844205f, -0.05939626f, -0.020390417f, +0.01374377f, +0.044976473f, +0.07873563f, +0.12207942f, +0.18412720f,
    -0.19048831f, -0.07587487f, -0.03220580f, -0.00011795067f, +0.02721784f, +0.04380719f, +0.07886723f, +0.13193911f,
    -0.13935551f, -0.092902906f, -0.052706074f, -0.017797327f, +0.015312965f, +0.056098964f, +0.11203423f, +0.24448302f,
    -0.17986591f, -0.10738580f, -0.06376371f, -0.026595421f, +0.00842492f, +0.04272362f, +0.08608052f, +0.15240218f,
    -0.10953678f, -0.057022586f, -0.012483291f, +0.024463262f, +0.06076792f, +0.09776234f, +0.12983681f, +0.18648379f,
    -0.16471463f, -0.089491285f, -0.037574016f, +0.004444791f, +0.039293647f, +0.07845859f, +0.12893885f, +0.23508036f
};

// 4-bit TCQ codebook (product_mono/iter100, 1024-state bitshift trellis). Decode: state_t = read_10_bits(qs, t*4)
static __constant__ float d_turbo4_tcq_codebook_fattn[1024] = {
    GGML_CUDA_TURBO4_TCQ_CODEBOOK_VALUES
};

// 2-bit TCQ codebook (product_mono/iter090, 256-state bitshift trellis). If you copy these, credit spiritbuun!
// CUDA GLA product-aware training, 100 iters on Qwen3.5-27B FWHT-rotated KV activations. Decode: state_t = read_8_bits(qs, t*2)
static __constant__ float d_turbo2_tcq_codebook_fattn[256] = {
    -0.18030643f, -0.11009848f, -0.04742626f, +0.02894132f, -0.10523465f, -0.031312924f, +0.031491395f, +0.12263535f,
    -0.15660362f, -0.055477407f, +0.0046675834f, +0.06166081f, -0.07506216f, -0.016963918f, +0.043737844f, +0.116496615f,
    -0.08632783f, -0.022493735f, +0.041032985f, +0.10660284f, -0.06274858f, -0.0036939639f, +0.02095157f, +0.07539709f,
    -0.09802641f, -0.008419088f, +0.059072323f, +0.17311879f, -0.093109086f, -0.02654333f, +0.014827672f, +0.07793592f,
    -0.031235758f, +0.01271591f, +0.08752262f, +0.17246453f, -0.14595252f, -0.07227624f, +0.013628688f, +0.08131674f,
    -0.036909282f, +0.0018896917f, +0.05209119f, +0.12407892f, -0.13689458f, -0.06054520f, +0.0064648795f, +0.07551241f,
    -0.18980840f, -0.110128626f, -0.046503957f, +0.026387159f, -0.034967307f, +0.04810357f, +0.072072044f, +0.14355458f,
    -0.10182410f, -0.02907887f, +0.014033012f, +0.083419636f, -0.056140676f, +0.008405868f, +0.066070884f, +0.14037225f,
    -0.117427245f, -0.047159385f, +0.016928354f, +0.08142885f, -0.029359628f, +0.045608785f, +0.10559447f, +0.20061271f,
    -0.040425077f, +0.029068163f, +0.08408973f, +0.13628258f, -0.16633821f, -0.10711727f, -0.04196669f, +0.027895834f,
    -0.0054065837f, +0.058898676f, +0.12688550f, +0.18268861f, -0.16287325f, -0.11218357f, -0.07165227f, -0.009524379f,
    -0.24026902f, -0.073219374f, -0.0005165726f, +0.05959821f, -0.05532953f, +0.027044486f, +0.09425678f, +0.15356481f,
    -0.14381111f, -0.10563502f, -0.037867088f, +0.023611993f, -0.03624307f, +0.049588434f, +0.12192037f, +0.23462485f,
    -0.14990251f, -0.09659304f, -0.05886742f, +0.014878461f, -0.009889551f, +0.06910514f, +0.12120181f, +0.22596690f,
    -0.08290075f, -0.009009629f, +0.066151775f, +0.12188313f, -0.11591514f, -0.06952189f, -0.031633306f, +0.023740824f,
    -0.20510401f, -0.103369795f, +0.09148037f, +0.17268716f, -0.16597997f, -0.09207068f, -0.032810967f, +0.024847647f,
    -0.02487482f, +0.049298953f, +0.09624215f, +0.14217524f, -0.18418685f, -0.10147012f, -0.05841265f, +0.008057022f,
    -0.14269894f, -0.092456274f, -0.026881337f, +0.049792137f, -0.019881032f, +0.030333601f, +0.09736802f, +0.17764080f,
    -0.19579841f, -0.114739306f, -0.026823774f, +0.07466014f, -0.09001050f, -0.041468445f, +0.028473806f, +0.08870695f,
    -0.019396419f, +0.042828932f, +0.10885327f, +0.13335012f, -0.15005013f, -0.074581385f, -0.028608415f, +0.03848942f,
    -0.09687270f, -0.057059396f, +0.0077843578f, +0.06302297f, -0.23247094f, -0.14509225f, -0.032651436f, +0.027010715f,
    -0.047595482f, +0.06280303f, +0.114691675f, +0.17124057f, -0.21092793f, -0.13704823f, -0.07340412f, +0.0039013291f,
    -0.062834196f, +0.012601906f, +0.012601906f, +0.08721347f, -0.13256435f, -0.024173854f, +0.07723171f, +0.14801070f,
    -0.06471605f, -0.0017903054f, -0.0017903054f, +0.058302354f, -0.09731802f, -0.03400696f, +0.02762442f, +0.08986137f,
    -0.08288722f, -0.019051429f, +0.045709886f, +0.15211061f, -0.09507891f, -0.015612489f, +0.025347246f, +0.087257534f,
    -0.066236064f, -0.0047936034f, +0.06386274f, +0.15401669f, -0.105809286f, -0.051802177f, +0.01073050f, +0.08292137f,
    -0.11884470f, -0.04404144f, +0.02550729f, +0.02550729f, -0.01731189f, +0.062161792f, +0.12127554f, +0.21981733f,
    -0.17066145f, -0.11660990f, -0.049425896f, +0.021293938f, -0.04711412f, +0.026577346f, +0.055197213f, +0.12541275f,
    -0.028268812f, +0.015206398f, +0.09002519f, +0.12699963f, -0.10059831f, -0.026676945f, +0.059903253f, +0.13054545f,
    -0.09582803f, -0.033371232f, +0.010346129f, +0.066766635f, -0.09964944f, -0.028686784f, +0.021184925f, +0.09120017f,
    -0.16957201f, -0.07594450f, +0.04172865f, +0.18313301f, -0.051526368f, +0.011877304f, +0.011877304f, +0.07956263f,
    -0.13432936f, -0.05269006f, +0.03536416f, +0.117640756f, -0.022776067f, +0.042032316f, +0.10472976f, +0.18042557f
};

// FWHT rotation sign arrays for FA inline rotation (same values as turbo-quant-cuda.cuh)
static __constant__ float d_turbo_wht_signs1_fattn[128] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f};
static __constant__ float d_turbo_wht_signs2_fattn[128] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f};

// InnerQ: per-channel inverse scale for Q pre-rotation (fattn compilation unit)
// Initialized to all 1.0 (identity). Updated by turbo_innerq_finalize_calibration().
static __device__ float d_innerq_channel_scale_inv_fattn[128];

// Q² calibration: accumulate per-position E[Q²] after FWHT rotation
// Used for product-aware TCQ codebook training (weight positions by query importance)
// Enabled by TURBO_Q_CALIBRATE=1 env var
static __device__ double d_q_channel_sq_fattn[128]; // sum of Q²ᵢ per position
static __device__ int    d_q_channel_count_fattn;    // token count
static __constant__ int  d_q_calibrate_fattn;        // 1 = accumulating (constant: fast broadcast read)

// CAS-based double atomicAdd for portability (MSVC/CUDA 12.x lacks the native overload)
static __device__ __forceinline__ double atomicAdd_double(double * addr, double val) {
    unsigned long long int * addr_ull = (unsigned long long int *)addr;
    unsigned long long int old = *addr_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_ull, assumed,
                        __double_as_longlong(__longlong_as_double(assumed) + val));
    } while (assumed != old);
    return __longlong_as_double(old);
}

#define FATTN_KQ_STRIDE       256
#define HALF_MAX_HALF         __float2half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.

// log(2) = 0.6931, by adding this to the KQ maximum used for the softmax the numerical range representable
//     by the VKQ accumulators is effectively being shifted up by a factor of 2.
// This reduces issues with numerical overflow but also causes larger values to be flushed to zero.
// However, as the output from FlashAttention will usually be used as an input for a matrix multiplication this should be negligible.
// Still, the value range should be shifted as much as necessary but as little as possible.
// The macro on the following line shifts it by a factor of 2**3=8, as was needed to fix https://github.com/ggml-org/llama.cpp/issues/18606 .
#define FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)

typedef void (* fattn_kernel_t)(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33);

typedef float (*vec_dot_KQ_t)(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds);

struct ggml_cuda_flash_attn_ext_f16_extra_data {
    uintptr_t K;
    uintptr_t V;
    uintptr_t end;
};

static inline ggml_cuda_flash_attn_ext_f16_extra_data ggml_cuda_flash_attn_ext_get_f16_extra_data(
        const ggml_tensor * dst, const bool need_f16_K, const bool need_f16_V) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    ggml_cuda_flash_attn_ext_f16_extra_data data = {};
    data.end = (uintptr_t) dst->data + ggml_nbytes(dst);

    if (need_f16_K && K->type != GGML_TYPE_F16) {
        data.end = GGML_PAD(data.end, 128);
        data.K   = data.end;
        data.end += ggml_nelements(K)*ggml_type_size(GGML_TYPE_F16);
    }

    if (need_f16_V && V->type != GGML_TYPE_F16) {
        if (V_is_K_view) {
            data.V = data.K;
        } else {
            data.end = GGML_PAD(data.end, 128);
            data.V   = data.end;
            data.end += ggml_nelements(V)*ggml_type_size(GGML_TYPE_F16);
        }
    }

    return data;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_f16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const half2 * K_h2 = (const half2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        __align__(16) half2 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_h2 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            ggml_cuda_mad(sum,                tmp[k_KQ_1] , ((const half2  *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#else
            ggml_cuda_mad(sum, __half22float2(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_bf16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const nv_bfloat162 * K_bf16 = (const nv_bfloat162 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        __align__(16) nv_bfloat162 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_bf16 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            // FIXME replace macros in vector FA kernel with templating and use FP32 for BF16
            ggml_cuda_mad(sum, ggml_cuda_cast<float2>(tmp[k_KQ_1]), __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]));
#else
            ggml_cuda_mad(sum, ggml_cuda_cast<float2>(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_0 * K_q4_0 = (const block_q4_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_0;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q4_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];
        sum += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x - (8/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_1 * K_q4_1 = (const block_q4_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q4_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q4_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_0 * K_q5_0 = (const block_q5_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_0;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q5_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += __half2float(K_q5_0[ib].d) * (sumi*Q_ds.x - (16/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q6_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q6_0 * K_q6_0 = (const block_q6_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI6_0;
        const int shift = k_KQ & (QI8_1/2);

        int vl;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&vl, K_q6_0[ib].qs + sizeof(int)*iqs4);
        vl = (vl >> shift) & 0x0F0F0F0F;

        int vh;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&vh, K_q6_0[ib].qh + sizeof(int)*(iqs4 % 2));
        vh = (vh >> (4*(iqs4/2) + shift/2)) & 0x03030303;

        const int v = vl | (vh << 4);
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += __half2float(K_q6_0[ib].d) * (sumi*Q_ds.x - (32/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q6_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q6_1 * K_q6_1 = (const block_q6_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI6_1;
        const int shift = k_KQ & (QI8_1/2);

        int vl;
        ggml_cuda_memcpy_1<sizeof(int)>(&vl, K_q6_1[ib].qs + sizeof(int)*iqs4);
        vl = (vl >> shift) & 0x0F0F0F0F;

        int vh;
        ggml_cuda_memcpy_1<sizeof(int)>(&vh, K_q6_1[ib].qh + sizeof(int)*(iqs4 % 2));
        vh = (vh >> (4*(iqs4/2) + shift/2)) & 0x03030303;

        const int v = vl | (vh << 4);
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q6_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

// 2-bit-plane types: q8 int iq of a block lives in qs int (iq%2), plane (iq/2)
template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q3_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q3_0 * K_q3_0 = (const block_q3_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib   = k_KQ / QI8_1;
        const int iq   = k_KQ % QI8_1;

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q3_0[ib].qs + sizeof(int)*(iq % 2));
        v = (v >> (2*(iq/2))) & 0x03030303;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int), 2>(&vh, K_q3_0[ib].qh);
            vh >>= 4*iq;

            v |= (vh <<  2) & 0x00000004; // 0 ->  2
            v |= (vh <<  9) & 0x00000400; // 1 -> 10
            v |= (vh << 16) & 0x00040000; // 2 -> 18
            v |= (vh << 23) & 0x04000000; // 3 -> 26
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += __half2float(K_q3_0[ib].d) * (sumi*Q_ds.x - (4.0f/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q3_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q3_1 * K_q3_1 = (const block_q3_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib   = k_KQ / QI8_1;
        const int iq   = k_KQ % QI8_1;

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q3_1[ib].qs + sizeof(int)*(iq % 2));
        v = (v >> (2*(iq/2))) & 0x03030303;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int)>(&vh, K_q3_1[ib].qh);
            vh >>= 4*iq;

            v |= (vh <<  2) & 0x00000004; // 0 ->  2
            v |= (vh <<  9) & 0x00000400; // 1 -> 10
            v |= (vh << 16) & 0x00040000; // 2 -> 18
            v |= (vh << 23) & 0x04000000; // 3 -> 26
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q3_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q2_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q2_0 * K_q2_0 = (const block_q2_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib   = k_KQ / QI8_1;
        const int iq   = k_KQ % QI8_1;

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q2_0[ib].qs + sizeof(int)*(iq % 2));
        v = (v >> (2*(iq/2))) & 0x03030303;

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += __half2float(K_q2_0[ib].d) * (sumi*Q_ds.x - (2.0f/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q2_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q2_1 * K_q2_1 = (const block_q2_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib   = k_KQ / QI8_1;
        const int iq   = k_KQ % QI8_1;

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q2_1[ib].qs + sizeof(int)*(iq % 2));
        v = (v >> (2*(iq/2))) & 0x03030303;

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q2_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_1 * K_q5_1 = (const block_q5_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_1;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q5_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q5_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q8_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q8_0 * K_q8_0 = (const block_q8_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib  = k_KQ / QI8_0;
        const int iqs = k_KQ % QI8_0;

        int v;
        ggml_cuda_memcpy_1<sizeof(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        const float2 * Q_ds = (const float2 *) Q_ds_v;
        const float Q_d = Q_ds[k_KQ_0/nthreads].x;

        sum += vec_dot_q8_0_q8_1_impl<float, 1>(&v, &Q_q8[k_KQ_0/nthreads], K_q8_0[ib].d, Q_d);
    }

    return sum;
}


template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_turbo2_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v,
    const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    const block_turbo2_0 * K_t2 = (const block_turbo2_0 *) K_c;
    GGML_UNUSED(Q_q8); GGML_UNUSED(Q_ds_v);
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;
    float sum = 0.0f;
    int prev_ib = -1;
    float cn[4];
#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        const int base_f2 = k_KQ_0 + (threadIdx.x % nthreads) * cpy_ne;
        const int elem0 = base_f2 * 2;
        const int ib = elem0 / QK_TURBO2;
        const int j_start = elem0 % QK_TURBO2;

        if (ib != prev_ib) {
            const float norm = __half2float(K_t2[ib].norm);
#pragma unroll
            for (int c = 0; c < 4; c++) {
                cn[c] = d_turbo_centroids_2bit_fattn[c] * norm;
            }
            prev_ib = ib;
        }

        const uint8_t qs_lo = K_t2[ib].qs[j_start / 4];
        const uint8_t qs_hi = K_t2[ib].qs[j_start / 4 + 1];

#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
            const int lj = k_KQ_1 * 2;
            const uint8_t qs_b0 = (lj < 4) ? qs_lo : qs_hi;
            const int idx0 = (qs_b0 >> ((lj % 4) * 2)) & 0x3;
            const int lj1 = lj + 1;
            const uint8_t qs_b1 = (lj1 < 4) ? qs_lo : qs_hi;
            const int idx1 = (qs_b1 >> ((lj1 % 4) * 2)) & 0x3;
#ifdef V_DOT2_F32_F16_AVAILABLE
            const float2 qf = __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#else
            const float2 qf = ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
#endif
            sum += cn[idx0] * qf.x + cn[idx1] * qf.y;
        }
    }
    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_turbo3_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v,
    const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    const block_turbo3_0 * K_t3 = (const block_turbo3_0 *) K_c;
    GGML_UNUSED(Q_q8); GGML_UNUSED(Q_ds_v);
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;
    float sum = 0.0f;
    int prev_ib = -1;
    float cn[8];
#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        const int base_f2 = k_KQ_0 + (threadIdx.x % nthreads) * cpy_ne;
        const int elem0 = base_f2 * 2;
        const int ib = elem0 / QK_TURBO3;
        const int j_start = elem0 % QK_TURBO3;

        if (ib != prev_ib) {
            const float norm = __half2float(K_t3[ib].norm);
#pragma unroll
            for (int c = 0; c < 8; c++) {
                cn[c] = d_turbo_centroids_3bit_fattn[c] * norm;
            }
            prev_ib = ib;
        }

        const uint8_t qs_lo = K_t3[ib].qs[j_start / 4];
        const uint8_t qs_hi = K_t3[ib].qs[j_start / 4 + 1];
        const uint8_t signs = K_t3[ib].signs[j_start / 8];

#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
            const int lj = k_KQ_1 * 2;
            int idx0, idx1;
            { const uint8_t qs_b = (lj < 4) ? qs_lo : qs_hi;
              const uint8_t low2 = (qs_b >> ((lj % 4) * 2)) & 0x3;
              const uint8_t hi1  = (signs >> lj) & 0x1;
              idx0 = low2 | (hi1 << 2); }
            { const int lj1 = lj + 1;
              const uint8_t qs_b = (lj1 < 4) ? qs_lo : qs_hi;
              const uint8_t low2 = (qs_b >> ((lj1 % 4) * 2)) & 0x3;
              const uint8_t hi1  = (signs >> lj1) & 0x1;
              idx1 = low2 | (hi1 << 2); }
#ifdef V_DOT2_F32_F16_AVAILABLE
            const float2 qf = __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#else
            const float2 qf = ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
#endif
            sum += cn[idx0] * qf.x + cn[idx1] * qf.y;
        }
    }
    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_turbo4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v,
    const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    const block_turbo4_0 * K_t4 = (const block_turbo4_0 *) K_c;
    GGML_UNUSED(Q_q8); GGML_UNUSED(Q_ds_v);
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;
    float sum = 0.0f;
    int prev_ib = -1;
    float cn[16];
#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        const int base_f2 = k_KQ_0 + (threadIdx.x % nthreads) * cpy_ne;
        const int elem0 = base_f2 * 2;
        const int ib = elem0 / QK_TURBO4;
        const int j_start = elem0 % QK_TURBO4;

        if (ib != prev_ib) {
            const float norm = __half2float(K_t4[ib].norm);
#pragma unroll
            for (int c = 0; c < 16; c++) {
                cn[c] = d_turbo_centroids_4bit_fattn[c] * norm;
            }
            prev_ib = ib;
        }

        // 4-bit indices: 2 per byte, simple nibble extraction
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
            const int lj = k_KQ_1 * 2;
            const int j0 = j_start + lj;
            const uint8_t byte0 = K_t4[ib].qs[j0 / 2];
            const uint8_t byte1 = K_t4[ib].qs[(j0 + 1) / 2];
            const uint8_t idx0 = (j0 & 1) ? (byte0 >> 4) : (byte0 & 0xF);
            const uint8_t idx1 = ((j0 + 1) & 1) ? (byte1 >> 4) : (byte1 & 0xF);
            const float k0 = cn[idx0];
            const float k1 = cn[idx1];
#ifdef V_DOT2_F32_F16_AVAILABLE
            const float2 qf = __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#else
            const float2 qf = ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
#endif
            sum += k0 * qf.x + k1 * qf.y;
        }
    }
    return sum;
}

// TCQ 4-bit K dot product: 10-bit state -> codebook lookup
template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_turbo4_tcq_cb(
    const char * __restrict__ K_c, const void * __restrict__ Q_v,
    const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v,
    const float * __restrict__ cb) {
    const block_turbo4_tcq * K_tcq = (const block_turbo4_tcq *) K_c;
    GGML_UNUSED(Q_q8); GGML_UNUSED(Q_ds_v);
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;
    float sum = 0.0f;
    int prev_ib = -1;
    float norm = 0.0f;
#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        const int base_f2 = k_KQ_0 + (threadIdx.x % nthreads) * cpy_ne;
        const int elem0 = base_f2 * 2;
        const int ib = elem0 / QK_TURBO4_TCQ;
        const int j_start = elem0 % QK_TURBO4_TCQ;

        if (ib != prev_ib) {
            norm = __half2float(K_tcq[ib].norm);
            prev_ib = ib;
        }

#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
            const int lj = k_KQ_1 * 2;
            const int t0 = j_start + lj;
            const int t1 = t0 + 1;
            const int bp0 = t0 * 4;
            const uint16_t raw0 = (uint16_t)K_tcq[ib].qs[bp0/8] | ((uint16_t)K_tcq[ib].qs[bp0/8 + 1] << 8);
            const float k0 = cb[(raw0 >> (bp0 % 8)) & 0x3FF] * norm;
            const int bp1 = t1 * 4;
            const uint16_t raw1 = (uint16_t)K_tcq[ib].qs[bp1/8] | ((uint16_t)K_tcq[ib].qs[bp1/8 + 1] << 8);
            const float k1 = cb[(raw1 >> (bp1 % 8)) & 0x3FF] * norm;
#ifdef V_DOT2_F32_F16_AVAILABLE
            const float2 qf = __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#else
            const float2 qf = ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
#endif
            sum += k0 * qf.x + k1 * qf.y;
        }
    }
    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_turbo4_tcq(
    const char * __restrict__ K_c, const void * __restrict__ Q_v,
    const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    return vec_dot_fattn_vec_KQ_turbo4_tcq_cb<D, nthreads>(K_c, Q_v, Q_q8, Q_ds_v, d_turbo4_tcq_codebook_fattn);
}

// TCQ 3-bit K dot product: 9-bit state -> codebook lookup
// Core implementation takes explicit codebook pointer for SMEM/constant flexibility
template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_turbo3_tcq_cb(
    const char * __restrict__ K_c, const void * __restrict__ Q_v,
    const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v,
    const float * __restrict__ cb) {
    const block_turbo3_tcq * K_tcq = (const block_turbo3_tcq *) K_c;
    GGML_UNUSED(Q_q8); GGML_UNUSED(Q_ds_v);
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;
    float sum = 0.0f;
    int prev_ib = -1;
    float norm = 0.0f;
#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        const int base_f2 = k_KQ_0 + (threadIdx.x % nthreads) * cpy_ne;
        const int elem0 = base_f2 * 2;
        const int ib = elem0 / QK_TURBO3_TCQ;
        const int j_start = elem0 % QK_TURBO3_TCQ;

        if (ib != prev_ib) {
            norm = __half2float(K_tcq[ib].norm);
            prev_ib = ib;
        }

#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
            const int lj = k_KQ_1 * 2;
            const int t0 = j_start + lj;
            const int t1 = t0 + 1;
            const int bp0 = t0 * 3;
            const uint16_t raw0 = (uint16_t)K_tcq[ib].qs[bp0/8] | ((uint16_t)K_tcq[ib].qs[bp0/8 + 1] << 8);
            const float k0 = cb[(raw0 >> (bp0 % 8)) & 0x1FF] * norm;
            const int bp1 = t1 * 3;
            const uint16_t raw1 = (uint16_t)K_tcq[ib].qs[bp1/8] | ((uint16_t)K_tcq[ib].qs[bp1/8 + 1] << 8);
            const float k1 = cb[(raw1 >> (bp1 % 8)) & 0x1FF] * norm;
#ifdef V_DOT2_F32_F16_AVAILABLE
            const float2 qf = __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#else
            const float2 qf = ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
#endif
            sum += k0 * qf.x + k1 * qf.y;
        }
    }
    return sum;
}

// Wrapper using __constant__ codebook (for function pointer dispatch)
template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_turbo3_tcq(
    const char * __restrict__ K_c, const void * __restrict__ Q_v,
    const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    return vec_dot_fattn_vec_KQ_turbo3_tcq_cb<D, nthreads>(K_c, Q_v, Q_q8, Q_ds_v, d_turbo3_tcq_codebook_fattn);
}

// TCQ 2-bit K dot product: 8-bit state → codebook lookup
template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_turbo2_tcq_cb(
    const char * __restrict__ K_c, const void * __restrict__ Q_v,
    const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v,
    const float * __restrict__ cb) {
    const block_turbo2_tcq * K_tcq = (const block_turbo2_tcq *) K_c;
    GGML_UNUSED(Q_q8); GGML_UNUSED(Q_ds_v);
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;
    float sum = 0.0f;
    int prev_ib = -1;
    float norm = 0.0f;
#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        const int base_f2 = k_KQ_0 + (threadIdx.x % nthreads) * cpy_ne;
        const int elem0 = base_f2 * 2;
        const int ib = elem0 / QK_TURBO2_TCQ;
        const int j_start = elem0 % QK_TURBO2_TCQ;

        if (ib != prev_ib) {
            norm = __half2float(K_tcq[ib].norm);
            prev_ib = ib;
        }

#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
            const int lj = k_KQ_1 * 2;
            const int t0 = j_start + lj;
            const int t1 = t0 + 1;
            const int bp0 = t0 * 2;
            const uint16_t raw0 = (uint16_t)K_tcq[ib].qs[bp0/8] | ((uint16_t)K_tcq[ib].qs[bp0/8 + 1] << 8);
            const float k0 = cb[(raw0 >> (bp0 % 8)) & 0xFF] * norm;
            const int bp1 = t1 * 2;
            const uint16_t raw1 = (uint16_t)K_tcq[ib].qs[bp1/8] | ((uint16_t)K_tcq[ib].qs[bp1/8 + 1] << 8);
            const float k1 = cb[(raw1 >> (bp1 % 8)) & 0xFF] * norm;
#ifdef V_DOT2_F32_F16_AVAILABLE
            const float2 qf = __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#else
            const float2 qf = ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
#endif
            sum += k0 * qf.x + k1 * qf.y;
        }
    }
    return sum;
}

// Wrapper using __constant__ codebook (for function pointer dispatch)
template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_turbo2_tcq(
    const char * __restrict__ K_c, const void * __restrict__ Q_v,
    const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    return vec_dot_fattn_vec_KQ_turbo2_tcq_cb<D, nthreads>(K_c, Q_v, Q_q8, Q_ds_v, d_turbo2_tcq_codebook_fattn);
}

template <typename Tds, int ni>
static __device__ __forceinline__ void quantize_q8_1_to_shared(
    const float * __restrict__ x, const float scale, int * __restrict__ yq32, void * __restrict__ yds) {

    float vals[sizeof(int)] = {0.0f};
#pragma unroll
    for (int l = 0; l < int(sizeof(int)); ++l) {
        vals[l] = (ni == WARP_SIZE || threadIdx.x < ni) ? scale * x[4*threadIdx.x + l] : 0.0f;
    }

    float amax = fabsf(vals[0]);
    float sum  = vals[0];
#pragma unroll
    for (int l = 1; l < int(sizeof(int)); ++l) {
        amax = fmaxf(amax, fabsf(vals[l]));
        sum += vals[l];
    }
#pragma unroll
    for (int mask = QI8_1/2; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFULL, amax, mask, 32));
        sum +=             __shfl_xor_sync(0xFFFFFFFFULL, sum,  mask, 32);
    }

    const float d = amax / 127;
    int q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    if (d != 0.0f) {
#pragma unroll
        for (int l = 0; l < int(sizeof(int)); ++l) {
            q8[l] = roundf(vals[l] / d);
        }
    }

    yq32[threadIdx.x] = q32;
    if (threadIdx.x % QI8_1 == 0 && (ni == WARP_SIZE || threadIdx.x < ni)) {
        if (std::is_same<Tds, half2>::value) {
            ((half2  *) yds)[threadIdx.x/QI8_1] =  make_half2(d, sum);
        } else {
            ((float2 *) yds)[threadIdx.x/QI8_1] = make_float2(d, sum);
        }
    }
}

typedef void (*dequantize_V_t)(const void *, void *, const int64_t);

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_f16(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    if constexpr (std::is_same_v<T, half>) {
        ggml_cuda_memcpy_1<ne*sizeof(half)>(dst, (const half *) vx + i0);
    } else if constexpr (std::is_same_v<T, float>) {
        static_assert(ne % 2 == 0, "bad ne");
        __align__(16) half2 tmp[ne/2];
        ggml_cuda_memcpy_1<ne*sizeof(half)>(tmp, (const half *) vx + i0);
        float2 * dst_f2 = (float2 *) dst;
#pragma unroll
        for (int l = 0; l < ne/2; ++l) {
            dst_f2[l] = __half22float2(tmp[l]);
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_bf16(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    static_assert(std::is_same_v<T, float>, "BF16 V dequantization only supports float output");
    static_assert(ne % 2 == 0, "bad ne");
    __align__(16) nv_bfloat162 tmp[ne/2];
    ggml_cuda_memcpy_1<ne*sizeof(nv_bfloat16)>(tmp, (const nv_bfloat16 *) vx + i0);
    float2 * dst_f2 = (float2 *) dst;
#pragma unroll
    for (int l = 0; l < ne/2; ++l) {
        dst_f2[l] = ggml_cuda_cast<float2>(tmp[l]);
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const int64_t ib    =  i0          /  QK4_0;
    const int     iqs   =  i0          % (QK4_0/2);
    const int     shift = (i0 % QK4_0) / (QK4_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;
    q = __vsubss4(q, 0x08080808);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const int64_t ib    =  i0          /  QK4_1;
    const int     iqs   =  i0          % (QK4_1/2);
    const int     shift = (i0 % QK4_1) / (QK4_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const int64_t ib    =  i0          /  QK5_0;
    const int     idq   =  i0          %  QK5_0;
    const int     iqs   =  i0          % (QK5_0/2);
    const int     shift = (i0 % QK5_0) / (QK5_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne, 2>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    q = __vsubss4(q, 0x10101010);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q6_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q6_0 * x = (const block_q6_0 *) vx;

    const int64_t ib    =  i0          /  QK6_0;
    const int     idq   =  i0          %  QK6_0;
    const int     iqs   =  i0          % (QK6_0/2);
    const int     shift = (i0 % QK6_0) / (QK6_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

#pragma unroll
    for (int l = 0; l < ne; ++l) {
        const int id = (idq + l) % QK6_0;
        const int j  = id % (QK6_0/2);
        const int h = (x[ib].qh[j % (QK6_0/4)] >> (4*(j / (QK6_0/4)))) & 0x0F;
        const int qh = id < QK6_0/2 ? (h & 0x03) : ((h >> 2) & 0x03);
        q |= qh << (8*l + 4);
    }

    q = __vsubss4(q, 0x20202020);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q6_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q6_1 * x = (const block_q6_1 *) vx;

    const int64_t ib    =  i0          /  QK6_1;
    const int     idq   =  i0          %  QK6_1;
    const int     iqs   =  i0          % (QK6_1/2);
    const int     shift = (i0 % QK6_1) / (QK6_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

#pragma unroll
    for (int l = 0; l < ne; ++l) {
        const int id = (idq + l) % QK6_1;
        const int j  = id % (QK6_1/2);
        const int h = (x[ib].qh[j % (QK6_1/4)] >> (4*(j / (QK6_1/4)))) & 0x0F;
        const int qh = id < QK6_1/2 ? (h & 0x03) : ((h >> 2) & 0x03);
        q |= qh << (8*l + 4);
    }

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

// 2-bit-plane types: elements idq..idq+ne-1 are bytes (idq%8)..(idq%8)+ne-1 of plane idq/8
template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q3_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q3_0 * x = (const block_q3_0 *) vx;

    const int64_t ib  = i0 / QK3_0;
    const int     idq = i0 % QK3_0;
    const int     jb  = idq % (QK3_0/4);
    const int     p   = idq / (QK3_0/4);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + jb);
    q >>= 2*p;
    q &= 0x03030303;

    {
        int qh;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 2);
        }
    }

    q = __vsubss4(q, 0x04040404);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q3_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q3_1 * x = (const block_q3_1 *) vx;

    const int64_t ib  = i0 / QK3_1;
    const int     idq = i0 % QK3_1;
    const int     jb  = idq % (QK3_1/4);
    const int     p   = idq / (QK3_1/4);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + jb);
    q >>= 2*p;
    q &= 0x03030303;

    {
        int qh;
        ggml_cuda_memcpy_1<sizeof(int)>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 2);
        }
    }

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q2_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q2_0 * x = (const block_q2_0 *) vx;

    const int64_t ib  = i0 / QK2_0;
    const int     idq = i0 % QK2_0;
    const int     jb  = idq % (QK2_0/4);
    const int     p   = idq / (QK2_0/4);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + jb);
    q >>= 2*p;
    q &= 0x03030303;

    q = __vsubss4(q, 0x02020202);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q2_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q2_1 * x = (const block_q2_1 *) vx;

    const int64_t ib  = i0 / QK2_1;
    const int     idq = i0 % QK2_1;
    const int     jb  = idq % (QK2_1/4);
    const int     p   = idq / (QK2_1/4);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + jb);
    q >>= 2*p;
    q &= 0x03030303;

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const int64_t ib    =  i0          /  QK5_1;
    const int     idq   =  i0          %  QK5_1;
    const int     iqs   =  i0          % (QK5_1/2);
    const int     shift = (i0 % QK5_1) / (QK5_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q8_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const int64_t ib  = i0 / QK8_0;
    const int     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_cuda_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same<T, half>::value) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same<T, float>::value) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * qs[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}


template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_turbo2_0(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_turbo2_0 * x = (const block_turbo2_0 *) vx;
    const int64_t ib = i0 / QK_TURBO2;
    const int     j0 = (int)(i0 % QK_TURBO2);
    const float norm = __half2float(x[ib].norm);
    static_assert(ne == 2 || ne == 4 || ne == 8, "bad ne");
    float cn[4];
#pragma unroll
    for (int c = 0; c < 4; c++) cn[c] = d_turbo_centroids_2bit_fattn[c] * norm;
    const uint8_t qs_lo = x[ib].qs[j0 / 4];
    const uint8_t qs_hi = (ne > 4 || j0 % 4 + ne > 4) ? x[ib].qs[j0 / 4 + 1] : 0;
    float vals[ne];
#pragma unroll
    for (int l = 0; l < ne; l++) {
        const int lj = j0 % 4 + l;
        const uint8_t qs_b = (lj < 4) ? qs_lo : qs_hi;
        const int idx = (qs_b >> ((lj % 4) * 2)) & 0x3;
        vals[l] = cn[idx];
    }
#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        for (int l0 = 0; l0 < ne; l0 += 2)
            ((half2 *)dst)[l0/2] = make_half2(__float2half(vals[l0]), __float2half(vals[l0+1]));
    } else
#endif
    if constexpr (std::is_same_v<T, float>) {
        for (int l = 0; l < ne; ++l) ((float *)dst)[l] = vals[l];
    } else { static_assert(std::is_same_v<T, void>, "bad type"); }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_turbo3_0(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_turbo3_0 * x = (const block_turbo3_0 *) vx;
    const int64_t ib = i0 / QK_TURBO3;
    const int     j0 = (int)(i0 % QK_TURBO3);
    const float norm = __half2float(x[ib].norm);
    static_assert(ne == 2 || ne == 4 || ne == 8, "bad ne");
    // Register-based centroid × norm LUT
    float cn[8];
#pragma unroll
    for (int c = 0; c < 8; c++) cn[c] = d_turbo_centroids_3bit_fattn[c] * norm;
    // Batch-load qs and signs bytes
    const uint8_t qs_lo = x[ib].qs[j0 / 4];
    const uint8_t qs_hi = (ne > 4 || j0 % 4 + ne > 4) ? x[ib].qs[j0 / 4 + 1] : 0;
    const uint8_t signs = x[ib].signs[j0 / 8];
    float vals[ne];
#pragma unroll
    for (int l = 0; l < ne; l++) {
        const int lj = j0 % 4 + l;
        const uint8_t qs_b = (lj < 4) ? qs_lo : qs_hi;
        const uint8_t low2 = (qs_b >> ((lj % 4) * 2)) & 0x3;
        const uint8_t hi1  = (signs >> ((j0 % 8) + l)) & 0x1;
        vals[l] = cn[low2 | (hi1 << 2)];
    }
#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        for (int l0 = 0; l0 < ne; l0 += 2)
            ((half2 *)dst)[l0/2] = make_half2(__float2half(vals[l0]), __float2half(vals[l0+1]));
    } else
#endif
    if constexpr (std::is_same_v<T, float>) {
        for (int l = 0; l < ne; ++l) ((float *)dst)[l] = vals[l];
    } else { static_assert(std::is_same_v<T, void>, "bad type"); }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_turbo4_0(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_turbo4_0 * x = (const block_turbo4_0 *) vx;
    const int64_t ib = i0 / QK_TURBO4;
    const int     j0 = (int)(i0 % QK_TURBO4);
    const float norm = __half2float(x[ib].norm);
    static_assert(ne == 2 || ne == 4 || ne == 8, "bad ne");
    float cn[16];
#pragma unroll
    for (int c = 0; c < 16; c++) cn[c] = d_turbo_centroids_4bit_fattn[c] * norm;
    float vals[ne];
#pragma unroll
    for (int l = 0; l < ne; l++) {
        const int j = j0 + l;
        const uint8_t idx = (j & 1) ? (x[ib].qs[j / 2] >> 4) : (x[ib].qs[j / 2] & 0xF);
        vals[l] = cn[idx];
    }
#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        for (int l0 = 0; l0 < ne; l0 += 2)
            ((half2 *)dst)[l0/2] = make_half2(__float2half(vals[l0]), __float2half(vals[l0+1]));
    } else
#endif
    if constexpr (std::is_same_v<T, float>) {
        for (int l = 0; l < ne; ++l) ((float *)dst)[l] = vals[l];
    } else { static_assert(std::is_same_v<T, void>, "bad type"); }
}

// TCQ decode-time V alpha: mirrors d_tcq_decode_alpha_v from fattn.cu
// When TURBO_TCQ_DECODE_ALPHA_V is set, this is loaded via fattn.cu's load_tcq_decode_alpha_fattn_common()
static __constant__ float d_tcq_decode_alpha_v_fattn = 1.0f;

// TCQ 4-bit V dequant: 10-bit state -> codebook lookup
template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_turbo4_tcq_cb(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0,
        const float * __restrict__ cb) {
    const block_turbo4_tcq * x = (const block_turbo4_tcq *) vx;
    const int64_t ib = i0 / QK_TURBO4_TCQ;
    const int     j0 = (int)(i0 % QK_TURBO4_TCQ);
    const float norm = __half2float(x[ib].norm) * d_tcq_decode_alpha_v_fattn;
    static_assert(ne == 2 || ne == 4 || ne == 8, "bad ne");
    float vals[ne];
#pragma unroll
    for (int l = 0; l < ne; l++) {
        const int t = j0 + l;
        const int bit_pos = t * 4;
        const uint16_t raw = (uint16_t)x[ib].qs[bit_pos/8] | ((uint16_t)x[ib].qs[bit_pos/8 + 1] << 8);
        const int state = (raw >> (bit_pos % 8)) & 0x3FF;
        vals[l] = cb[state] * norm;
    }
#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        for (int l0 = 0; l0 < ne; l0 += 2)
            ((half2 *)dst)[l0/2] = make_half2(__float2half(vals[l0]), __float2half(vals[l0+1]));
    } else
#endif
    if constexpr (std::is_same_v<T, float>) {
        for (int l = 0; l < ne; ++l) ((float *)dst)[l] = vals[l];
    } else { static_assert(std::is_same_v<T, void>, "bad type"); }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_turbo4_tcq(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    dequantize_V_turbo4_tcq_cb<T, ne>(vx, dst, i0, d_turbo4_tcq_codebook_fattn);
}

// TCQ 3-bit V dequant: 9-bit state -> codebook lookup
// Core implementation takes explicit codebook pointer for SMEM/constant flexibility
template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_turbo3_tcq_cb(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0,
        const float * __restrict__ cb) {
    const block_turbo3_tcq * x = (const block_turbo3_tcq *) vx;
    const int64_t ib = i0 / QK_TURBO3_TCQ;
    const int     j0 = (int)(i0 % QK_TURBO3_TCQ);
    const float norm = __half2float(x[ib].norm) * d_tcq_decode_alpha_v_fattn;
    static_assert(ne == 2 || ne == 4 || ne == 8, "bad ne");
    float vals[ne];
#pragma unroll
    for (int l = 0; l < ne; l++) {
        const int t = j0 + l;
        const int bit_pos = t * 3;
        const uint16_t raw = (uint16_t)x[ib].qs[bit_pos/8] | ((uint16_t)x[ib].qs[bit_pos/8 + 1] << 8);
        const int state = (raw >> (bit_pos % 8)) & 0x1FF;
        vals[l] = cb[state] * norm;
    }
#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        for (int l0 = 0; l0 < ne; l0 += 2)
            ((half2 *)dst)[l0/2] = make_half2(__float2half(vals[l0]), __float2half(vals[l0+1]));
    } else
#endif
    if constexpr (std::is_same_v<T, float>) {
        for (int l = 0; l < ne; ++l) ((float *)dst)[l] = vals[l];
    } else { static_assert(std::is_same_v<T, void>, "bad type"); }
}

// Wrapper using __constant__ codebook (for function pointer dispatch via dequantize_V_t)
template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_turbo3_tcq(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    dequantize_V_turbo3_tcq_cb<T, ne>(vx, dst, i0, d_turbo3_tcq_codebook_fattn);
}

// TCQ 2-bit V dequant: 8-bit state → codebook lookup
template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_turbo2_tcq_cb(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0,
        const float * __restrict__ cb) {
    const block_turbo2_tcq * x = (const block_turbo2_tcq *) vx;
    const int64_t ib = i0 / QK_TURBO2_TCQ;
    const int     j0 = (int)(i0 % QK_TURBO2_TCQ);
    const float norm = __half2float(x[ib].norm) * d_tcq_decode_alpha_v_fattn;
    static_assert(ne == 2 || ne == 4 || ne == 8, "bad ne");
    float vals[ne];
#pragma unroll
    for (int l = 0; l < ne; l++) {
        const int t = j0 + l;
        const int bit_pos = t * 2;
        const uint16_t raw = (uint16_t)x[ib].qs[bit_pos/8] | ((uint16_t)x[ib].qs[bit_pos/8 + 1] << 8);
        const int state = (raw >> (bit_pos % 8)) & 0xFF;
        vals[l] = cb[state] * norm;
    }
#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        for (int l0 = 0; l0 < ne; l0 += 2)
            ((half2 *)dst)[l0/2] = make_half2(__float2half(vals[l0]), __float2half(vals[l0+1]));
    } else
#endif
    if constexpr (std::is_same_v<T, float>) {
        for (int l = 0; l < ne; ++l) ((float *)dst)[l] = vals[l];
    } else { static_assert(std::is_same_v<T, void>, "bad type"); }
}

// Wrapper using __constant__ codebook (for function pointer dispatch via dequantize_V_t)
template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_turbo2_tcq(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    dequantize_V_turbo2_tcq_cb<T, ne>(vx, dst, i0, d_turbo2_tcq_codebook_fattn);
}

template <ggml_type type_K, int D, int nthreads>
constexpr __device__ vec_dot_KQ_t get_vec_dot_KQ() {
    if constexpr (type_K == GGML_TYPE_F16) {
        return vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_0) {
        return vec_dot_fattn_vec_KQ_q4_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_1) {
        return vec_dot_fattn_vec_KQ_q4_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_0) {
        return vec_dot_fattn_vec_KQ_q5_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_1) {
        return vec_dot_fattn_vec_KQ_q5_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q6_0) {
        return vec_dot_fattn_vec_KQ_q6_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q6_1) {
        return vec_dot_fattn_vec_KQ_q6_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q3_0) {
        return vec_dot_fattn_vec_KQ_q3_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q3_1) {
        return vec_dot_fattn_vec_KQ_q3_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q2_0) {
        return vec_dot_fattn_vec_KQ_q2_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q2_1) {
        return vec_dot_fattn_vec_KQ_q2_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q8_0) {
        return vec_dot_fattn_vec_KQ_q8_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_BF16) {
        return vec_dot_fattn_vec_KQ_bf16<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TURBO2_0) {
        return vec_dot_fattn_vec_KQ_turbo2_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TURBO3_0) {
        return vec_dot_fattn_vec_KQ_turbo3_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TURBO4_0) {
        return vec_dot_fattn_vec_KQ_turbo4_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TURBO4_TCQ) {
        return vec_dot_fattn_vec_KQ_turbo4_tcq<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TURBO3_TCQ) {
        return vec_dot_fattn_vec_KQ_turbo3_tcq<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TURBO2_TCQ) {
        return vec_dot_fattn_vec_KQ_turbo2_tcq<D, nthreads>;
    } else {
        static_assert(type_K == -1, "bad type");
        return nullptr;
    }
}

template <ggml_type type_V, typename T, int ne>
constexpr __device__ dequantize_V_t get_dequantize_V() {
    if constexpr (type_V == GGML_TYPE_F16) {
        return dequantize_V_f16<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_0) {
        return dequantize_V_q4_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_1) {
        return dequantize_V_q4_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_0) {
        return dequantize_V_q5_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_1) {
        return dequantize_V_q5_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q6_0) {
        return dequantize_V_q6_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q6_1) {
        return dequantize_V_q6_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q3_0) {
        return dequantize_V_q3_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q3_1) {
        return dequantize_V_q3_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q2_0) {
        return dequantize_V_q2_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q2_1) {
        return dequantize_V_q2_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q8_0) {
        return dequantize_V_q8_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_BF16) {
        return dequantize_V_bf16<float, ne>;
    } else if constexpr (type_V == GGML_TYPE_TURBO2_0) {
        return dequantize_V_turbo2_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_TURBO3_0) {
        return dequantize_V_turbo3_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_TURBO4_0) {
        return dequantize_V_turbo4_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_TURBO4_TCQ) {
        return dequantize_V_turbo4_tcq<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_TURBO3_TCQ) {
        return dequantize_V_turbo3_tcq<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_TURBO2_TCQ) {
        return dequantize_V_turbo2_tcq<T, ne>;
    } else {
        static_assert(type_V == -1, "bad type");
        return nullptr;
    }
}

template <int ncols1>
__launch_bounds__(FATTN_KQ_STRIDE/2, 1)
static __global__ void flash_attn_mask_to_KV_max(
        const half2 * mask_ptr, int * KV_max_ptr, const int ne30, const int64_t s31, const int64_t s33) {
    const half2 * GGML_CUDA_RESTRICT mask   = mask_ptr;
    int         * GGML_CUDA_RESTRICT KV_max = KV_max_ptr;

    const int ne31     = gridDim.x;
    const int tid      = threadIdx.x;
    const int sequence = blockIdx.y;
    const int jt       = blockIdx.x;

    mask += sequence*s33 + jt*ncols1*s31;

    __shared__ int buf_iw[WARP_SIZE];
    if (tid < WARP_SIZE) {
        buf_iw[tid] = 1;
    }
    ggml_cuda_pdl_sync();
    __syncthreads();

    int KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    for (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        int all_inf = 1;

#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            const float2 tmp = __half22float2(mask[j*s31 + KV_max_sj/2 + tid]);
            all_inf = all_inf && int(isinf(tmp.x)) && int(isinf(tmp.y));
        }

        all_inf = warp_reduce_all(all_inf);
        if (tid % WARP_SIZE == 0) {
            buf_iw[tid / WARP_SIZE] = all_inf;
        }
        __syncthreads();
        all_inf = buf_iw[tid % WARP_SIZE];
        __syncthreads();
        all_inf = warp_reduce_all(all_inf);

        if (!all_inf) {
            break;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    if (threadIdx.x != 0) {
        return;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}


template<int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup_uniform(
        float * dst_ptr,
        const float2 * dst_fixup_ptr,
        const int ne01, const int ne02,
        const int ne12, const int nblocks_stream_k,
        const int gqa_ratio,
        const int blocks_per_tile,
        const uint3 fd_iter_j_z_ne12,
        const uint3 fd_iter_j_z,
        const uint3 fd_iter_j) {
    constexpr int ncols = ncols1*ncols2;
    ggml_cuda_pdl_lc();
    float        * GGML_CUDA_RESTRICT dst       = dst_ptr;
    const float2 * GGML_CUDA_RESTRICT dst_fixup = dst_fixup_ptr;

    const int tile_idx = blockIdx.x; // One block per output tile.
    const int j        = blockIdx.y;
    const int c        = blockIdx.z;
    const int jc       = j*ncols2 + c;
    const int tid      = threadIdx.x;

    // nblocks_stream_k is a multiple of ntiles_dst (== gridDim.x), so each tile gets the same number of blocks.
    const int b_first = tile_idx * blocks_per_tile;
    const int b_last  = b_first + blocks_per_tile - 1;

    const float * dst_fixup_data = ((const float *) dst_fixup) + nblocks_stream_k*(2*2*ncols);

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const uint2 dm0 = fast_div_modulo(tile_idx, fd_iter_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y,    fd_iter_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y,    fd_iter_j);

    const int sequence = dm0.x;
    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm2.y;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        return;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    ggml_cuda_pdl_sync();
    // Load the partial result that needs a fixup
    float dst_val = *dst;
    float max_val;
    float rowsum;
    {
        const float2 tmp = dst_fixup[b_last*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Combine with all previous blocks in this tile.
    for (int bidx = b_last - 1; bidx >= b_first; --bidx) {
        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(nblocks_stream_k + bidx)*ncols + jc];

        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

// General fixup kernel for the case where the number of blocks per tile is not uniform across tiles
// (blocks_num.x not a multiple of ntiles_dst)
template <int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup_general(
        float * dst_ptr,
        const float2 * dst_fixup_ptr,
        const int ne01, const int ne02,
        const int gqa_ratio,
        const int total_work,
        const uint3 fd_iter_k_j_z_ne12,
        const uint3 fd_iter_k_j_z,
        const uint3 fd_iter_k_j,
        const uint3 fd_iter_k) {
    float        * GGML_CUDA_RESTRICT dst       = dst_ptr;
    const float2 * GGML_CUDA_RESTRICT dst_fixup = dst_fixup_ptr;
    constexpr int ncols = ncols1*ncols2;

    const int bidx0 = blockIdx.x;
    const int j     = blockIdx.y;
    const int c     = blockIdx.z;
    const int jc    = j*ncols2 + c;
    const int tid   = threadIdx.x;

    const float * dst_fixup_data = ((const float *) dst_fixup) + gridDim.x*(2*2*ncols);

    const int kbc0      = int64_t(bidx0 + 0)*total_work / gridDim.x;
    const int kbc0_stop = int64_t(bidx0 + 1)*total_work / gridDim.x;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = fastmodulo(kbc0, fd_iter_k) == 0;
    const bool did_not_write_last      = fastdiv(kbc0, fd_iter_k) == fastdiv(kbc0_stop, fd_iter_k) && fastmodulo(kbc0_stop, fd_iter_k) != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const uint2 dm0 = fast_div_modulo(kbc0, fd_iter_k_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y, fd_iter_k_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y, fd_iter_k_j);
    const uint2 dm3 = fast_div_modulo(dm2.y, fd_iter_k);

    const int sequence = dm0.x;
    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm3.x;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        return;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    float dst_val = 0.0f;
    float max_val = 0.0f;
    float rowsum  = 0.0f;
    ggml_cuda_pdl_sync();
    {
        dst_val = *dst;

        const float2 tmp = dst_fixup[bidx0*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Iterate over previous blocks and compute the combined results.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    const int tile_kbc0 = fastdiv(kbc0, fd_iter_k);
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        const int kbc = int64_t(bidx)*total_work / gridDim.x;
        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(gridDim.x + bidx)*ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (fastmodulo(kbc, fd_iter_k) == 0 || fastdiv(kbc, fd_iter_k) < tile_kbc0) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

template<int D> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_combine_results(
        const float  * VKQ_parts_ptr,
        const float2 * VKQ_meta_ptr,
        float * dst_ptr,
        const int parallel_blocks) {
    ggml_cuda_pdl_lc();
    const float  * GGML_CUDA_RESTRICT VKQ_parts = VKQ_parts_ptr;
    const float2 * GGML_CUDA_RESTRICT VKQ_meta  = VKQ_meta_ptr;
    float        * GGML_CUDA_RESTRICT dst       = dst_ptr;
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    const int ne01 = gridDim.x;
    const int ne02 = gridDim.y;

    const int col      = blockIdx.x;
    const int head     = blockIdx.y;
    const int sequence = blockIdx.z;

    const int j_dst_unrolled = (sequence*ne01 + col)*ne02 + head;

    VKQ_parts += j_dst_unrolled * parallel_blocks*D;
    VKQ_meta  += j_dst_unrolled * parallel_blocks;
    dst       += j_dst_unrolled *                 D;

    const int tid = threadIdx.x;
    __builtin_assume(tid < D);

    extern __shared__ float2 meta[];
    ggml_cuda_pdl_sync();
    for (int i = tid; i < 2*parallel_blocks; i += D) {
        ((float *) meta)[i] = ((const float *)VKQ_meta) [i];
    }

    __syncthreads();

    float kqmax = meta[0].x;
    for (int l = 1; l < parallel_blocks; ++l) {
        kqmax = max(kqmax, meta[l].x);
    }

    float VKQ_numerator   = 0.0f;
    float VKQ_denominator = 0.0f;
    for (int l = 0; l < parallel_blocks; ++l) {
        const float KQ_max_scale = expf(meta[l].x - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y;
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

template <int DV, int ncols1, int ncols2>
void launch_fattn(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst, fattn_kernel_t fattn_kernel, const int nwarps, const size_t nbytes_shared,
    const int nbatch_fa, const bool need_f16_K, const bool need_f16_V, const bool stream_k, const int warp_size = WARP_SIZE
) {
    constexpr int ncols = ncols1 * ncols2;

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_size(K));
    GGML_ASSERT(V->nb[0] == ggml_element_size(V));

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t main_stream = ctx.stream();
    const int id  = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(KQV, need_f16_K, need_f16_V);

    ggml_cuda_pool_alloc<int>    KV_max(pool);
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    const char * K_data = (const char *) K->data;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    const char * V_data = (const char *) V->data;
    size_t nb21 = V->nb[1];
    size_t nb22 = V->nb[2];
    size_t nb23 = V->nb[3];

    if (need_f16_K && K->type != GGML_TYPE_F16) {
        const size_t bs = ggml_blck_size(K->type);
        const size_t ts = ggml_type_size(K->type);

        GGML_ASSERT(f16_extra.K != 0);
        half * K_f16 = (half *) f16_extra.K;
        if (ggml_is_contiguously_allocated(K)) {
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(K->type);
            to_fp16(K_data, K_f16, ggml_nelements(K), main_stream);

            nb11 = nb11*bs*sizeof(half)/ts;
            nb12 = nb12*bs*sizeof(half)/ts;
            nb13 = nb13*bs*sizeof(half)/ts;
        } else {
            GGML_ASSERT(K->nb[0] == ts);
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(K->type);
            const int64_t s01 = nb11 / ts;
            const int64_t s02 = nb12 / ts;
            const int64_t s03 = nb13 / ts;
            to_fp16(K_data, K_f16, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_stream);

            nb11 = K->ne[0] * sizeof(half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_data = (char *) K_f16;
    }

    if (need_f16_V && V->type != GGML_TYPE_F16) {
        if (V_is_K_view) {
            V_data = K_data;
            nb21   = nb11;
            nb22   = nb12;
            nb23   = nb13;
        } else {
            const size_t bs = ggml_blck_size(V->type);
            const size_t ts = ggml_type_size(V->type);

            GGML_ASSERT(f16_extra.V != 0);
            half * V_f16 = (half *) f16_extra.V;
            if (ggml_is_contiguously_allocated(V)) {
                to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(V->type);
                to_fp16(V_data, V_f16, ggml_nelements(V), main_stream);
                V_data = (char *) V_f16;

                nb21 = nb21*bs*sizeof(half)/ts;
                nb22 = nb22*bs*sizeof(half)/ts;
                nb23 = nb23*bs*sizeof(half)/ts;
            } else {
                GGML_ASSERT(V->nb[0] == ts);
                to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(V->type);
                const int64_t s01 = nb21 / ts;
                const int64_t s02 = nb22 / ts;
                const int64_t s03 = nb23 / ts;
                to_fp16(V_data, V_f16, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_stream);

                nb21 = V->ne[0] * sizeof(half);
                nb22 = V->ne[1] * nb21;
                nb23 = V->ne[2] * nb22;
            }
            V_data = (char *) V_f16;
        }
    }

    const int ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    const int gqa_ratio    = Q->ne[2] / K->ne[2];
    const int ntiles_z_gqa = ((gqa_ratio + ncols2 - 1) / ncols2);
    const int ntiles_dst   = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    if (mask && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
        const int64_t s31 = mask->nb[1] / sizeof(half2);
        const int64_t s33 = mask->nb[3] / sizeof(half2);

        const dim3 blocks_num_KV_max(ntiles_x, Q->ne[3], 1);
        const dim3 block_dim_KV_max(FATTN_KQ_STRIDE/2, 1, 1);

        const int ne_KV_max = blocks_num_KV_max.x*blocks_num_KV_max.y;
        const int iter_k = K->ne[1] / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_KV_max, block_dim_KV_max, 0, main_stream);
        ggml_cuda_kernel_launch(flash_attn_mask_to_KV_max<ncols1>, launch_params,
            (const half2 *) mask->data, KV_max.ptr, iter_k, s31, s33);
        CUDA_CHECK(cudaGetLastError());
    }

    const dim3 block_dim(warp_size, nwarps, 1);
    int max_blocks_per_sm = 1; // Max. number of active blocks limited by occupancy.
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fattn_kernel, block_dim.x * block_dim.y * block_dim.z, nbytes_shared));
    GGML_ASSERT(max_blocks_per_sm > 0);
    int parallel_blocks = max_blocks_per_sm;

    const int ntiles_KV = (K->ne[1] + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by KV cache length.

    dim3 blocks_num;
    if (stream_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        const int max_blocks = max_blocks_per_sm*nsm;
        const int tiles_nwaves = (ntiles_dst + max_blocks - 1) / max_blocks;
        const int tiles_efficiency_percent = 100 * ntiles_dst / (max_blocks*tiles_nwaves);

        const bool use_stream_k = cc >= GGML_CUDA_CC_ADA_LOVELACE || amd_wmma_available(cc) || tiles_efficiency_percent < 75;

        blocks_num.x = ntiles_dst;
        blocks_num.y = 1;
        blocks_num.z = 1;

        if(use_stream_k) {
            const int nblocks_stream_k_raw = std::min(max_blocks, ntiles_KV*ntiles_dst);
            // Round down to a multiple of ntiles_dst so that each output tile gets the same number of blocks (avoids fixup).
            // Only do this if the occupancy loss from rounding is acceptable.
            const int nblocks_stream_k_rounded = (nblocks_stream_k_raw / ntiles_dst) * ntiles_dst;
            const int max_efficiency_loss_percent = 5;
            const int efficiency_loss_percent = nblocks_stream_k_rounded > 0
                ? 100 * (nblocks_stream_k_raw - nblocks_stream_k_rounded) / nblocks_stream_k_raw
                : 100;
            const int nblocks_stream_k = efficiency_loss_percent <= max_efficiency_loss_percent
                ? nblocks_stream_k_rounded
                : nblocks_stream_k_raw;

            blocks_num.x = nblocks_stream_k;
        }

        if (ntiles_dst % blocks_num.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((size_t(blocks_num.x) * ncols * (2 + DV/2)));
        }
    } else {
        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_blocks = std::min(parallel_blocks, ntiles_KV);

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        const int blocks_per_wave = nsm * max_blocks_per_sm;
        int nwaves_best = 0;
        int efficiency_percent_best = 0;
        for (int parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KV; ++parallel_blocks_test) {
            const int nblocks_total = ntiles_dst * parallel_blocks_test;
            const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
            const int efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            if (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
                break;
            }

            if (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

        blocks_num.x = ntiles_x;
        blocks_num.y = parallel_blocks;
        blocks_num.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];

        if (parallel_blocks > 1) {
            dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
        }
    }

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    GGML_ASSERT(block_dim.x % warp_size == 0);

        ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num, block_dim, nbytes_shared, main_stream);
        ggml_cuda_kernel_launch(fattn_kernel, launch_params,
        (const char *) Q->data,
        K_data,
        V_data,
        mask ? ((const char *) mask->data) : nullptr,
        sinks ? ((const char *) sinks->data) : nullptr,
        KV_max.ptr,
        !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (float *) KQV->data, dst_tmp_meta.ptr,
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        Q->ne[0], ne01,     Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
        K->ne[0], K->ne[1], K->ne[2], K->ne[3], nb11, nb12, nb13,
        nb21, nb22, nb23,
        mask ? mask->ne[1] : 0, mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
        mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0
    );
    CUDA_CHECK(cudaGetLastError());

    if (stream_k) {
        if ((int)blocks_num.x % ntiles_dst == 0 && (int)blocks_num.x > ntiles_dst) {
            // Optimized fixup: nblocks_stream_k is a multiple of ntiles_dst, launch one block per tile.
            const int nblocks_sk  = (int)blocks_num.x;
            const int bpt         = nblocks_sk / ntiles_dst;

            const uint3 fd0 = init_fastdiv_values(ntiles_x * ntiles_z_gqa * K->ne[2]);
            const uint3 fd1 = init_fastdiv_values(ntiles_x * ntiles_z_gqa);
            const uint3 fd2 = init_fastdiv_values(ntiles_x);

            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {(unsigned)ntiles_dst, ncols1, ncols2};

            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, 0, main_stream);
            ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_uniform<DV, ncols1, ncols2>, launch_params,
                (float *) KQV->data, dst_tmp_meta.ptr,
                 Q->ne[1], Q->ne[2], K->ne[2], nblocks_sk,
                 gqa_ratio, bpt, fd0, fd1, fd2);
        } else if (ntiles_dst % blocks_num.x != 0) {
            // General fixup for the cases where nblocks_stream_k < ntiles_dst.
            const int total_work = ntiles_KV * ntiles_dst;

            const uint3 fd_k_j_z_ne12 = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa * K->ne[2]);
            const uint3 fd_k_j_z      = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa);
            const uint3 fd_k_j        = init_fastdiv_values(ntiles_KV * ntiles_x);
            const uint3 fd_k          = init_fastdiv_values(ntiles_KV);

            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {blocks_num.x, ncols1, ncols2};

            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, 0, main_stream);
            ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_general<DV, ncols1, ncols2>, launch_params,
                (float *) KQV->data, dst_tmp_meta.ptr,
                 Q->ne[1], Q->ne[2], gqa_ratio, total_work,
                 fd_k_j_z_ne12, fd_k_j_z, fd_k_j, fd_k);
        }
    } else if (parallel_blocks > 1) {
        const dim3 block_dim_combine(DV, 1, 1);
        const dim3 blocks_num_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        const size_t nbytes_shared_combine = parallel_blocks*sizeof(float2);

        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, nbytes_shared_combine, main_stream);
        ggml_cuda_kernel_launch(flash_attn_combine_results<DV>, launch_params,
            dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data, parallel_blocks);
    }
    CUDA_CHECK(cudaGetLastError());
}
