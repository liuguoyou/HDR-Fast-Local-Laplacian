#include "fllf.h"
#include <omp.h>

static const unsigned char hdr_weights[256] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 127,
126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98,
97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62,
61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26,
25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };
/*
static const float g_green[256] = { -4.495188567858563f, -4.376902513824406f, -4.258616459790245f, -4.140330405756109f, -4.022044351722f,
-3.903766473016247f, -3.785547869485901f, -3.667789300052836f, -3.551086595145196f, -3.436443977173664f, -3.325015236888474f, -3.217456992113802f,
-3.113965559539528f, -3.014726021015882f, -2.919869682341458f, -2.829899534716569f, -2.745142383458001f, -2.665612551770399f, -2.591088334999103f,
-2.521223959666765f, -2.455638040972245f, -2.393937998479167f, -2.335706727051395f, -2.280478859631515f, -2.227791072411007f, -2.177242254491476f,
-2.128582200748065f, -2.081592130588677f, -2.036078430150703f, -1.991921757663193f, -1.949147418988918f, -1.907838707310797f, -1.867967173164509f,
-1.82941743403727f, -1.792079196995946f, -1.755962771057843f, -1.721079154224879f, -1.687422632879139f, -1.654924568076226f, -1.623502552784026f,
-1.593128088531818f, -1.563828166581599f, -1.535444773377969f, -1.507828087942594f, -1.480793539376492f, -1.454172353227419f, -1.427836927920927f,
-1.401732295679401f, -1.37575393723451f, -1.349860662445201f, -1.324046353051506f, -1.298326317000189f, -1.272758238248259f, -1.247442878212544f,
-1.222534187351445f, -1.198174644643426f, -1.174496192288205f, -1.151621081967457f, -1.129631546734315f, -1.108549860024433f, -1.088337097443043f,
-1.068967506886988f, -1.050397638917214f, -1.032545900744024f, -1.01533337950493f, -0.9985966138610081f, -0.9821931349342679f, -0.9659626535259955f,
-0.9497852760683522f, -0.9335480348733147f, -0.9171391616315456f, -0.9004788268484102f, -0.8835003410404735f, -0.8661305116060189f, -0.848300248430941f,
-0.8299772056760178f, -0.811130914822457f, -0.7917444008674416f, -0.7718590949879803f, -0.7515966110548782f, -0.7309894820569609f, -0.7100659232696552f,
-0.6888436568014721f, -0.6673438812223302f, -0.645624045491676f, -0.6237757768369305f, -0.6018945948542633f, -0.5800735301568039f, -0.5584544075789513f,
-0.5372399336028693f, -0.5165682638315363f, -0.4965378063506874f, -0.4772425362868266f, -0.4588109724326463f, -0.4413411644911276f, -0.4248803351492261f,
-0.4094163204785795f, -0.3949372417790737f, -0.3813102756217481f, -0.3684200811256404f, -0.3561621885807329f, -0.3444161840745683f, -0.3330479535717519f,
-0.3219322740906242f, -0.3109814149696172f, -0.3001060326232261f, -0.2891943522628176f, -0.2781540572342218f, -0.2669259558543653f, -0.2554501400408598f,
-0.243679635153109f, -0.2316393252330284f, -0.2193589408548617f, -0.2068750324958519f, -0.1942229001386251f, -0.1814090770763475f, -0.1684398957855264f,
-0.1553320497868689f, -0.1421016870196481f, -0.1287208238424733f, -0.1152014142211266f, -0.101522180753788f, -0.08767181558481973f, -0.07363940156440968f,
-0.05939599025907261f, -0.04491358696996217f, -0.03016511625292084f, -0.01518317860296658f, -2.987161629164348e-010f, 0.01535585455083899f, 0.03085811841309066f,
0.04646192624422962f, 0.06211629925816364f, 0.07776858065913483f, 0.09336437332886716f, 0.1088696161660568f, 0.1242508266013549f, 0.1394734046606767f,
0.1545015903352507f, 0.1692984190038875f, 0.1838256748021809f, 0.1980438418267614f, 0.2119120530555998f, 0.2253880368578245f, 0.238428060958243f,
0.2510035443845262f, 0.2631399817605737f, 0.2748254312991421f, 0.28610880887382f, 0.297037263282879f, 0.3076824356841423f, 0.3181291022546697f,
0.3284952388327596f, 0.3389019037788978f, 0.3494610593836137f, 0.3602588150778692f, 0.3713497299088395f, 0.3828040624192637f, 0.3947239572526446f,
0.4071811184582126f, 0.4202124981761335f, 0.4337972658276763f, 0.4479132773251115f, 0.4625373030604675f, 0.4776559139349011f, 0.4932641037542644f,
0.5092641942802232f, 0.525598094113605f, 0.5422049551201327f, 0.5590210272675598f, 0.5759795043536228f, 0.5930513142983744f, 0.6102317809772161f,
0.6275053540837148f, 0.6449113669681764f, 0.6624396429636334f, 0.6800810066821195f, 0.6979566812187769f, 0.7161520775396826f, 0.7346359751309774f,
0.7533753887764382f, 0.7723354597419969f, 0.791479338984089f, 0.8107620351232172f, 0.830135968987339f, 0.8495978090823741f, 0.8689652768439686f,
0.8881747429444479f, 0.9071408918636873f, 0.9257856445160277f, 0.9440262656169104f, 0.961830888000559f, 0.9791726957613827f, 0.9960233765859706f,
1.012390702541488f, 1.028286122788909f, 1.043722939670784f, 1.058707208784036f, 1.073254338052837f, 1.087383620352974f, 1.101172019613268f,
1.114702424286893f, 1.128064153365038f, 1.141282048874175f, 1.154346744193161f, 1.167201483080542f, 1.179975693981454f, 1.192809653631343f,
1.205855520164373f, 1.219278485009415f, 1.233171424219742f, 1.247635188784494f, 1.262779416638736f, 1.27867615268749f, 1.295368083668602f,
1.31284853344622f, 1.331109155932061f, 1.350139701211712f, 1.369927745664234f, 1.39045837962871f, 1.41166214936591f, 1.433460337781512f,
1.455763666163052f, 1.478420255508945f, 1.501327046200211f, 1.52449445186896f, 1.547933741769982f, 1.571657146563662f, 1.595677981766039f,
1.619942557469689f, 1.644387884963924f, 1.668896033223513f, 1.693328378959243f, 1.717570559958985f, 1.741487261543596f, 1.764867004572483f,
1.788039635270343f, 1.811331913831321f, 1.834910078934323f, 1.858939805916749f, 1.883480804847351f, 1.908442906949248f, 1.933704043787625f,
1.959273703191545f, 1.985160514114911f, 2.011371286393902f, 2.037909346965433f, 2.064771584505865f, 2.091943007041721f, 2.119386212127226f,
2.147019607494937f, 2.174721206745915f, 2.202489539377733f, 2.230283821969566f, 2.258107298266413f, 2.285930774563247f };
*/
static const float g_green[256] = { -4.79521232f, -4.64313555f, -4.49105878f, -4.33898202f, -4.18924636f, -4.04163174f, -3.89867436f, -3.76074174f, 
-3.6231151f, -3.48475572f, -3.35485605f, -3.23833689f, -3.13334311f, -3.03987577f, -2.95455127f, -2.87540838f, -2.79944266f, -2.72519429f, -2.65157048f, 
-2.57864728f, -2.50865802f, -2.44107164f, -2.37544608f, -2.31147327f, -2.2508349f, -2.19466882f, -2.14427483f, -2.10018426f, -2.06093606f, -2.02526993f, 
-1.99312225f, -1.96207728f, -1.92958241f, -1.89449451f, -1.85604715f, -1.81576853f, -1.77532836f, -1.73526914f, -1.69600921f, -1.65903331f, -1.62565094f, 
-1.59561193f, -1.56865231f, -1.54311634f, -1.51740751f, -1.49081124f, -1.46291938f, -1.43487315f, -1.40783422f, -1.38252309f, -1.3594531f, -1.33863938f, 
-1.31971099f, -1.30162566f, -1.28400552f, -1.26712445f, -1.2505613f, -1.23278577f, -1.21259915f, -1.19039085f, -1.16633404f, -1.14094057f, -1.11443689f, 
-1.08724202f, -1.06102421f, -1.03642717f, -1.01405488f, -0.993449995f, -0.974425962f, -0.956806353f, -0.940424228f, -0.925185663f, -0.910487041f, -0.895757602f, 
-0.880457445f, -0.864415787f, -0.848012564f, -0.831325121f, -0.814440276f, -0.797679604f, -0.779912601f, -0.760415819f, -0.739078357f, -0.716505949f, 
-0.693797451f, -0.672000054f, -0.653097519f, -0.635859571f, -0.619601549f, -0.604207388f, -0.590112587f, -0.576491764f, -0.563216972f, -0.550165686f,
-0.537507129f, -0.524969501f, -0.511572289f, -0.497743952f, -0.483540965f, -0.469017601f, -0.454324795f, -0.439739337f, -0.424894662f, -0.409453173f, 
-0.393407383f, -0.377111755f, -0.360276173f, -0.341477861f, -0.321317508f, -0.300412603f, -0.279511208f, -0.259334425f, -0.240577491f, -0.223283044f, 
-0.207637818f, -0.19345707f, -0.180576242f, -0.168109317f, -0.155144258f, -0.141483737f, -0.126920093f, -0.111729364f, -0.095907672f, -0.0797796079f, 
-0.0636593341f, -0.0473358334f, -0.0312869001f, -0.0153946114f, 3.34576236e-11f, 0.0150054835f, 0.0291425687f, 0.0422614419f, 0.0542075505f, 0.0650391863f, 
0.0746164691f, 0.0833476363f, 0.0922011085f, 0.101091908f, 0.109540613f, 0.1173225f, 0.124396738f, 0.13075432f, 0.137089271f, 0.143767314f, 0.151321286f, 
0.159937469f, 0.169798508f, 0.181093601f, 0.193753437f, 0.207893644f, 0.223415463f, 0.239667727f, 0.257043894f, 0.275952637f, 0.296895739f, 0.319976674f, 
0.344164922f, 0.368239603f, 0.390965961f, 0.411782993f, 0.429298409f, 0.445077873f, 0.460579205f, 0.475543665f, 0.489701267f, 0.502770223f, 0.51521849f, 
0.527023082f, 0.537954723f, 0.547773709f, 0.556644581f, 0.565288155f, 0.575682136f, 0.586815076f, 0.598864073f, 0.612014797f, 0.626461975f, 0.642222864f, 
0.658527376f, 0.67361984f, 0.686347924f, 0.6968778f, 0.705104036f, 0.710916208f, 0.715914605f, 0.721791269f, 0.729530291f, 0.741252499f, 0.756394756f, 
0.774358283f, 0.794510698f, 0.816179064f, 0.838646992f, 0.861783576f, 0.88441188f, 0.905711152f, 0.926384535f, 0.946313342f, 0.964740508f, 0.981692696f, 
0.996088691f, 1.00738253f, 1.01498918f, 1.01790387f, 1.01955349f, 1.0236399f, 1.03256474f, 1.04568129f, 1.06228419f, 1.08160364f, 1.10279884f, 1.12495076f, 
1.14784282f, 1.17122099f, 1.19515074f, 1.21986071f, 1.24560201f, 1.27265082f, 1.30156058f, 1.33248393f, 1.36498069f, 1.39811603f, 1.4308408f, 1.46197557f, 
1.48701837f, 1.50531439f, 1.51852695f, 1.52856883f, 1.53667197f, 1.54370289f, 1.55013338f, 1.55664406f, 1.56403341f, 1.57324152f, 1.58507984f, 1.59992376f, 
1.61821243f, 1.64030702f, 1.66639329f, 1.69627765f, 1.72968546f, 1.76623259f, 1.80538533f, 1.84640225f, 1.8882483f, 1.92946415f, 1.97103114f, 2.01440658f, 
2.06181098f, 2.11308849f, 2.1670955f, 2.22306425f, 2.27817175f, 2.33095778f, 2.38338555f, 2.43581332f };

static const float pyramid_filter[25] = { 0.0025000f, 0.0125000f, 0.0200000f, 0.0125000f, 0.0025000f,
0.0125000f, 0.0625000f, 0.1000000f, 0.0625000f, 0.0125000f, 0.0200000f, 0.1000000f, 0.1600000f, 0.1000000f, 0.0200000f,
0.0125000f, 0.0625000f, 0.1000000f, 0.0625000f, 0.0125000f, 0.0025000f, 0.0125000f, 0.0200000f, 0.0125000f, 0.0025000f };

void filter_downsample(const float* src, float* dst, const IM_SIZE_TYPE* level_sizes, const int level){
    const int srcw = level_sizes[level - 1].w;
    const int srch = level_sizes[level - 1].h;
    const int dstw = level_sizes[level].w;
    const int dsth = level_sizes[level].h;
    const unsigned char wodd = srcw & 1;
    const unsigned char hodd = srch & 1;
    int i = 0;
    int j = 0;
    const float* srcptr = src;
    float* dstptr = dst;
    const int srcwx2 = srcw << 1;
    int side_residue = 0;
    float side_weight = 0.0f;
    //top left, w = 3
    for (i = 0; i < 3; ++i){
        *dstptr += srcptr[i] * pyramid_filter[12 + i];
        *dstptr += srcptr[i + srcw] * pyramid_filter[12 + i + 5];
        *dstptr += srcptr[i + srcwx2] * pyramid_filter[12 + i + 10];
    }
    *dstptr /= 0.49f;
    srcptr += 2;
    ++dstptr;
    //top middle, w = 5
    for (j = 0; j < dstw - 2; j++){
        for (i = -2; i < 3; ++i){
            *dstptr += srcptr[i] * pyramid_filter[12 + i];
            *dstptr += srcptr[i + srcw] * pyramid_filter[12 + i + 5];
            *dstptr += srcptr[i + srcwx2] * pyramid_filter[12 + i + 10];
        }
        *dstptr /= 0.7f;
        srcptr += 2;
        ++dstptr;
    }
    //top right
    if (wodd == 0){ //even width
        side_residue = 2;
        side_weight = 0.665f;
    }
    else{ //odd width
        side_residue = 1;
        side_weight = 0.49f;
    }
    for (i = -2; i < side_residue; ++i){
        *dstptr += srcptr[i] * pyramid_filter[12 + i];
        *dstptr += srcptr[i + srcw] * pyramid_filter[12 + i + 5];
        *dstptr += srcptr[i + srcwx2] * pyramid_filter[12 + i + 10];
    }
    *dstptr /= side_weight;
    
    srcptr = src;
    dstptr = dst;
    //middle
    for (j = 1; j <= dsth - 2; j++){
        srcptr = src + j * srcwx2;
        dstptr = dst + j * dstw;
        //left, w = 3
        for (i = 0; i < 3; ++i){
            *dstptr += srcptr[i - srcwx2] * pyramid_filter[12 + i - 10];
            *dstptr += srcptr[i - srcw] * pyramid_filter[12 + i - 5];
            *dstptr += srcptr[i] * pyramid_filter[12 + i];
            *dstptr += srcptr[i + srcw] * pyramid_filter[12 + i + 5];
            *dstptr += srcptr[i + srcwx2] * pyramid_filter[12 + i + 10];
        }
        *dstptr /= 0.7f;
        srcptr += 2;
        ++dstptr;
        //middle, w = 5
        int k = 0;
        for (k = 0; k < dstw - 2; k++){
            for (i = -2; i < 3; ++i){
                *dstptr += srcptr[i - srcwx2] * pyramid_filter[12 + i - 10];
                *dstptr += srcptr[i - srcw] * pyramid_filter[12 + i - 5];
                *dstptr += srcptr[i] * pyramid_filter[12 + i];
                *dstptr += srcptr[i + srcw] * pyramid_filter[12 + i + 5];
                *dstptr += srcptr[i + srcwx2] * pyramid_filter[12 + i + 10];
            }
            srcptr += 2;
            ++dstptr;
        }
        //right
        if (wodd == 0){ //even width
            side_residue = 2;
            side_weight = 0.95f;
        }
        else{ //odd width
            side_residue = 1;
            side_weight = 0.7f;
        }
        for (i = -2; i < side_residue; ++i){
            *dstptr += srcptr[i - srcwx2] * pyramid_filter[12 + i - 10];
            *dstptr += srcptr[i - srcw] * pyramid_filter[12 + i - 5];
            *dstptr += srcptr[i] * pyramid_filter[12 + i];
            *dstptr += srcptr[i + srcw] * pyramid_filter[12 + i + 5];
            *dstptr += srcptr[i + srcwx2] * pyramid_filter[12 + i + 10];
        }
        *dstptr /= side_weight;
    }
    srcptr = src + srcwx2 * (dsth - 1);
    dstptr = dst + dstw * (dsth - 1);
    //bottom left, w = 3
    for (i = 0; i < 3; ++i){
        *dstptr += srcptr[i - srcwx2] * pyramid_filter[12 + i - 10];
        *dstptr += srcptr[i - srcw] * pyramid_filter[12 + i - 5];
        *dstptr += srcptr[i] * pyramid_filter[12 + i];
    }
    if (hodd == 0){ //even height
        for (i = 0; i < 3; ++i){
            *dstptr += srcptr[i + srcw] * pyramid_filter[12 + i + 5];
        }
        side_weight = 0.665f;
    }
    else{ //odd height
        side_weight = 0.49f;
    }
    *dstptr /= side_weight;
    srcptr += 2;
    ++dstptr;
    //bottom middle, w = 5
    for (j = 0; j < dstw - 2; j++){
        for (i = -2; i < 3; ++i){
            *dstptr += srcptr[i - srcwx2] * pyramid_filter[12 + i - 10];
            *dstptr += srcptr[i - srcw] * pyramid_filter[12 + i - 5];
            *dstptr += srcptr[i] * pyramid_filter[12 + i];
        }
        if (hodd == 0){ //even height
            for (i = -2; i < 3; ++i){
                *dstptr += srcptr[i + srcw] * pyramid_filter[12 + i + 5];
            }
            side_weight = 0.95f;
        }
        else{ //odd height
            side_weight = 0.7f;
        }
        *dstptr /= side_weight;
        srcptr += 2;
        ++dstptr;
    }
    //bottom right
    if (wodd == 0){ //even width
        side_residue = 2;
        if (hodd == 0){ //even height
            side_weight = 0.9025f;
            for (i = -2; i < side_residue; ++i){
                *dstptr += srcptr[i + srcw] * pyramid_filter[12 + i + 5];
            }
        }
        else{ //odd height
            side_weight = 0.665f;
        }
    }
    else{ //odd width
        side_residue = 1;
        if (hodd == 0){ //even height
            side_weight = 0.665f;
            for (i = -2; i < side_residue; ++i){
                *dstptr += srcptr[i + srcw] * pyramid_filter[12 + i + 5];
            }
        }
        else{ //odd height
            side_weight = 0.49f;
        }
    }
    for (i = -2; i < side_residue; ++i){
        *dstptr += srcptr[i - srcwx2] * pyramid_filter[12 + i - 10];
        *dstptr += srcptr[i - srcw] * pyramid_filter[12 + i - 5];
        *dstptr += srcptr[i] * pyramid_filter[12 + i];
    }
    *dstptr /= side_weight;
}

void filter_upsample(const float* src, float* dst, basic_algebra_ptr op, const IM_SIZE_TYPE* level_sizes, const int level){
    const int srcw = level_sizes[level + 1].w;
    const int srch = level_sizes[level + 1].h;
    const int dstw = level_sizes[level].w;
    const int dsth = level_sizes[level].h;
    const unsigned char wodd = dstw & 1;
    const unsigned char hodd = dsth & 1;
    int i = 0;
    int j = 0;
    const float* srcptr = src;
    float* dstptr = dst;
    float temp = 0.0f;
    //top left
    temp += *srcptr * pyramid_filter[12];
    temp += *(srcptr + 1) * pyramid_filter[14];
    temp += *(srcptr + srcw) * pyramid_filter[22];
    temp += *(srcptr + srcw + 1) * pyramid_filter[24];
    temp /= 0.2025f;
    *dstptr = op(*dstptr, temp);
    //top middle, zero column
    dstptr = dst + 1;
    for (i = 0; i < srcw - 1; i++){
        float temp = 0.0f;
        temp += *srcptr * pyramid_filter[11];
        temp += *(srcptr + 1) * pyramid_filter[13];
        temp += *(srcptr + srcw) * pyramid_filter[21];
        temp += *(srcptr + srcw + 1) * pyramid_filter[23];
        temp /= 0.225f;
        *dstptr = op(*dstptr, temp);
        ++srcptr;
        dstptr += 2;
    }
    //top middle, src column
    srcptr = src + 1;
    dstptr = dst + 2;
    for (i = 0; i < srcw - 2; i++){
        float temp = 0.0f;
        temp += *srcptr * pyramid_filter[12];
        temp += *(srcptr + 1) * pyramid_filter[14];
        temp += *(srcptr - 1) * pyramid_filter[10];
        temp += *(srcptr + srcw) * pyramid_filter[22];
        temp += *(srcptr + srcw + 1) * pyramid_filter[24];
        temp += *(srcptr + srcw - 1) * pyramid_filter[20];
        temp /= 0.225f;
        *dstptr = op(*dstptr, temp);
        ++srcptr;
        dstptr += 2;
    }
    //top right
    temp = 0.0f;
    srcptr = src + srcw - 1;
    if (wodd == 0){
        dstptr = dst + dstw - 2;
    }
    else{
        dstptr = dst + dstw - 1;
    }  
    temp += *srcptr * pyramid_filter[12];
    temp += *(srcptr - 1) * pyramid_filter[10];
    temp += *(srcptr + srcw) * pyramid_filter[22];
    temp += *(srcptr + srcw - 1) * pyramid_filter[20];
    temp /= 0.2025f;
    *dstptr = op(*dstptr, temp);
    if (wodd == 0){
        dstptr = dst + dstw - 1;
        temp = 0.0f;
        temp += *srcptr * pyramid_filter[11];
        temp += *(srcptr + srcw) * pyramid_filter[21];
        temp /= 0.1125f;
        *dstptr = op(*dstptr, temp);
    }
    //middle zero row
    const float* srcrow_start = src;
    float* dstrow_start = dst + dstw;
    for (j = 0; j < srch - 1; j++){
        float temp = 0.0f;
        srcptr = srcrow_start;
        dstptr = dstrow_start;
        //left
        temp += *srcptr * pyramid_filter[7];
        temp += *(srcptr + 1) * pyramid_filter[9];
        temp += *(srcptr + srcw) * pyramid_filter[17];
        temp += *(srcptr + srcw + 1) * pyramid_filter[19];
        temp /= 0.225f;
        *dstptr = op(*dstptr, temp);
        //middle, zero column
        ++dstptr;
        for (i = 0; i < srcw - 1; i++){
            float temp = 0.0f;
            temp += *srcptr * pyramid_filter[6];
            temp += *(srcptr + 1) * pyramid_filter[8];
            temp += *(srcptr + srcw) * pyramid_filter[16];
            temp += *(srcptr + srcw + 1) * pyramid_filter[18];
            temp /= 0.25f;
            *dstptr = op(*dstptr, temp);
            ++srcptr;
            dstptr += 2;
        }
        if (wodd == 0){
            //right
            temp = 0.0f;
            temp += *srcptr * pyramid_filter[6];
            temp += *(srcptr + srcw) * pyramid_filter[16];
            temp /= 0.125f;
            *dstptr = op(*dstptr, temp);
        }
        //middle, src column
        srcptr = srcrow_start + 1;
        dstptr = dstrow_start + 2;
        for (i = 0; i < srcw - 2; i++){
            float temp = 0.0f;
            temp += *srcptr * pyramid_filter[7];
            temp += *(srcptr + 1) * pyramid_filter[9];
            temp += *(srcptr - 1) * pyramid_filter[5];
            temp += *(srcptr + srcw) * pyramid_filter[17];
            temp += *(srcptr + srcw + 1) * pyramid_filter[19];
            temp += *(srcptr + srcw - 1) * pyramid_filter[15];
            temp /= 0.25f;
            *dstptr = op(*dstptr, temp);
            ++srcptr;
            dstptr += 2;
        }
        //right
        temp = 0.0f;
        temp += *srcptr * pyramid_filter[7];
        temp += *(srcptr - 1) * pyramid_filter[5];
        temp += *(srcptr + srcw) * pyramid_filter[17];
        temp += *(srcptr + srcw - 1) * pyramid_filter[15];
        temp /= 0.225f;
        *dstptr = op(*dstptr, temp);
        srcrow_start += srcw;
        dstrow_start += (dstw << 1);
    }
    //middle src row
    srcrow_start = src + srcw;
    dstrow_start = dst + (dstw << 1);
    for (j = 0; j < srch - 2; j++){
        float temp = 0.0f;
        srcptr = srcrow_start;
        dstptr = dstrow_start;
        //left
        temp += *srcptr * pyramid_filter[12];
        temp += *(srcptr + 1) * pyramid_filter[14];
        temp += *(srcptr + srcw) * pyramid_filter[22];
        temp += *(srcptr + srcw + 1) * pyramid_filter[24];
        temp += *(srcptr - srcw) * pyramid_filter[2];
        temp += *(srcptr - srcw + 1) * pyramid_filter[4];
        temp /= 0.225f;
        *dstptr = op(*dstptr, temp);
        //middle, zero column
        ++dstptr;
        for (i = 0; i < srcw - 1; i++){
            float temp = 0.0f;
            temp += *srcptr * pyramid_filter[11];
            temp += *(srcptr + 1) * pyramid_filter[13];
            temp += *(srcptr + srcw) * pyramid_filter[21];
            temp += *(srcptr + srcw + 1) * pyramid_filter[23];
            temp += *(srcptr - srcw) * pyramid_filter[1];
            temp += *(srcptr - srcw + 1) * pyramid_filter[3];
            temp /= 0.25f;
            *dstptr = op(*dstptr, temp);
            ++srcptr;
            dstptr += 2;
        }
        if (wodd == 0){
            //right
            temp = 0.0f;
            temp += *srcptr * pyramid_filter[11];
            temp += *(srcptr + srcw) * pyramid_filter[21];
            temp += *(srcptr - srcw) * pyramid_filter[1];
            temp /= 0.125f;
            *dstptr = op(*dstptr, temp);
        }
        //middle, src column
        srcptr = srcrow_start + 1;
        dstptr = dstrow_start + 2;
        for (i = 0; i < srcw - 2; i++){
            float temp = 0.0f;
            temp += *srcptr * pyramid_filter[12];
            temp += *(srcptr + 1) * pyramid_filter[14];
            temp += *(srcptr - 1) * pyramid_filter[10];
            temp += *(srcptr + srcw) * pyramid_filter[22];
            temp += *(srcptr - srcw) * pyramid_filter[2];
            temp += *(srcptr + srcw + 1) * pyramid_filter[24];
            temp += *(srcptr + srcw - 1) * pyramid_filter[20];
            temp += *(srcptr - srcw + 1) * pyramid_filter[4];
            temp += *(srcptr - srcw - 1) * pyramid_filter[0];
            temp /= 0.25f;
            *dstptr = op(*dstptr, temp);
            ++srcptr;
            dstptr += 2;
        }
        //right
        temp = 0.0f;
        temp += *srcptr * pyramid_filter[12];
        temp += *(srcptr - 1) * pyramid_filter[10];
        temp += *(srcptr + srcw) * pyramid_filter[22];
        temp += *(srcptr - srcw) * pyramid_filter[2];
        temp += *(srcptr + srcw - 1) * pyramid_filter[20];
        temp += *(srcptr - srcw - 1) * pyramid_filter[0];
        temp /= 0.225f;
        *dstptr = op(*dstptr, temp);
        srcrow_start += srcw;
        dstrow_start += (dstw << 1);
    }
    //2nd bottom left
    srcrow_start = src + srcw * (srch - 1);
    if (hodd == 0){
        dstrow_start = dst + dstw * (dsth - 2);
    }
    else{
        dstrow_start = dst + dstw * (dsth - 1);
    }
    srcptr = srcrow_start;
    dstptr = dstrow_start;
    temp = 0.0f;
    temp += *srcptr * pyramid_filter[12];
    temp += *(srcptr + 1) * pyramid_filter[14];
    temp += *(srcptr - srcw) * pyramid_filter[2];
    temp += *(srcptr - srcw + 1) * pyramid_filter[4];
    temp /= 0.2025f;
    *dstptr = op(*dstptr, temp);
    //2nd bottom middle, zero column
    dstptr = dstrow_start + 1;
    for (i = 0; i < srcw - 1; i++){
        float temp = 0.0f;
        temp += *srcptr * pyramid_filter[11];
        temp += *(srcptr + 1) * pyramid_filter[13];
        temp += *(srcptr - srcw) * pyramid_filter[1];
        temp += *(srcptr - srcw + 1) * pyramid_filter[3];
        temp /= 0.225f;
        *dstptr = op(*dstptr, temp);
        ++srcptr;
        dstptr += 2;
    }
    if (wodd == 0){
        //2nd bottom right
        temp = 0.0f;
        temp += *srcptr * pyramid_filter[11];
        temp += *(srcptr - srcw) * pyramid_filter[1];
        temp /= 0.1125f;
        *dstptr = op(*dstptr, temp);
    }
    //2nd bottom middle, src column
    srcptr = srcrow_start + 1;
    dstptr = dstrow_start + 2;
    for (i = 0; i < srcw - 2; i++){
        float temp = 0.0f;
        temp += *srcptr * pyramid_filter[12];
        temp += *(srcptr + 1) * pyramid_filter[14];
        temp += *(srcptr - 1) * pyramid_filter[10];
        temp += *(srcptr - srcw) * pyramid_filter[2];
        temp += *(srcptr - srcw + 1) * pyramid_filter[4];
        temp += *(srcptr - srcw - 1) * pyramid_filter[0];
        temp /= 0.225f;
        *dstptr = op(*dstptr, temp);
        ++srcptr;
        dstptr += 2;
    }
    //2nd bottom right
    temp = 0.0f;
    temp += *srcptr * pyramid_filter[12];
    temp += *(srcptr - 1) * pyramid_filter[10];
    temp += *(srcptr - srcw) * pyramid_filter[2];
    temp += *(srcptr - srcw - 1) * pyramid_filter[0];
    temp /= 0.2025f;
    *dstptr = op(*dstptr, temp);
    if (hodd == 0){
        //1st bottom left
        dstrow_start = dst + dstw * (dsth - 1);
        srcptr = srcrow_start;
        dstptr = dstrow_start;
        temp = 0.0f;
        temp += *srcptr * pyramid_filter[7];
        temp += *(srcptr + 1) * pyramid_filter[9];
        temp /= 0.1125f;
        *dstptr = op(*dstptr, temp);
        //1st bottom middle, zero column
        dstptr = dstrow_start + 1;
        for (i = 0; i < srcw - 1; i++){
            float temp = 0.0f;
            temp += *srcptr * pyramid_filter[6];
            temp += *(srcptr + 1) * pyramid_filter[8];
            temp /= 0.125f;
            *dstptr = op(*dstptr, temp);
            ++srcptr;
            dstptr += 2;
        }
        if (wodd == 0){
            //1st bottom right
            temp = 0.0f;
            temp += *srcptr * pyramid_filter[6];
            temp /= 0.0625f;
            *dstptr = op(*dstptr, temp);
        }
        //1st bottom middle, src column
        srcptr = srcrow_start + 1;
        dstptr = dstrow_start + 2;
        for (i = 0; i < srcw - 2; i++){
            float temp = 0.0f;
            temp += *srcptr * pyramid_filter[7];
            temp += *(srcptr + 1) * pyramid_filter[9];
            temp += *(srcptr - 1) * pyramid_filter[5];
            temp /= 0.125f;
            *dstptr = op(*dstptr, temp);
            ++srcptr;
            dstptr += 2;
        }
        //1st bottom right
        temp = 0.0f;
        temp += *srcptr * pyramid_filter[7];
        temp += *(srcptr - 1) * pyramid_filter[5];
        temp /= 0.1125f;
        *dstptr = op(*dstptr, temp);
    }
}

float** build_gaussian_pyramid(float* src, const int LLF_LEVEL, const IM_SIZE_TYPE* level_sizes){
    float** pyr = (float**)alloc_from_stack(LLF_LEVEL * sizeof(float*));
    pyr[0] = (float*)alloc_from_stack(IMG_WIDTH * IMG_HEIGHT * sizeof(float));
    memcpy(pyr[0], src, IMG_WIDTH * IMG_HEIGHT * sizeof(float));
    int level = 0;
    for (level = 1; level < LLF_LEVEL; ++level){
        pyr[level] = (float*)alloc_from_stack(level_sizes[level].w * level_sizes[level].h * sizeof(float));
        filter_downsample(pyr[level - 1], pyr[level], level_sizes, level);
    }
    return pyr;
}

void build_laplacian_pyramid(float** pyr, const int LLF_LEVEL, const IM_SIZE_TYPE* level_sizes){
    int level = 0;
    basic_algebra_ptr op = fminus; //minus the upsample result
    for (level = 0; level < LLF_LEVEL - 1; ++level){
        memset(pyr[level + 1], 0, level_sizes[level + 1].w * level_sizes[level + 1].h * sizeof(float)); //clean
        filter_downsample(pyr[level], pyr[level + 1], level_sizes, level + 1);
        filter_upsample(pyr[level + 1], pyr[level], op, level_sizes, level);
    }
}

float** alloc_empty_pyramid(const int LLF_LEVEL, IM_SIZE_TYPE* level_sizes){
    float** pyr = (float**)alloc_from_stack(LLF_LEVEL * sizeof(float*));
    int level = 0;
    int current_width = IMG_WIDTH;
    int current_height = IMG_HEIGHT;
    for (level = 0; level < LLF_LEVEL; ++level){
        pyr[level] = (float*)alloc_from_stack(current_width * current_height * sizeof(float));
        level_sizes[level].w = current_width;
        level_sizes[level].h = current_height;
        current_width = (current_width + 1) >> 1;
        current_height = (current_height + 1) >> 1;
    }
    return pyr;
}

void remap_lum(const float* src, float* dst, const float ref){
    const int IMG_SIZE = IMG_WIDTH * IMG_HEIGHT;
    const double NOISE_LEVEL = 0.01;
    const double LLF_SIGMA_R_RECIPROCAL = 1.0 / LLF_SIGMA_R;
    int idx = 0;
#pragma omp parallel for
    for (idx = 0; idx < IMG_SIZE; ++idx){
        float diff = src[idx] - ref;
        double absdiff = fabs(diff);
        char signdiff = (diff > 0) - (diff < 0);
        unsigned char is_edge = (absdiff > LLF_SIGMA_R);
        double ratiodiff = absdiff * LLF_SIGMA_R_RECIPROCAL;
        double smooth_term = max(0.0, min(1.0, 100.0 * absdiff - 1.0));
        double tau = smooth_term * smooth_term * (smooth_term - 2.0) * (smooth_term - 2.0);
        float edge = (float)(ref + signdiff * (LLF_BETA * (absdiff - LLF_SIGMA_R) + LLF_SIGMA_R));
        float detail = (float)(ref + signdiff * LLF_SIGMA_R * (pow(ratiodiff, LLF_ALPHA) * tau + ratiodiff * (1 - tau)));
        dst[idx] = is_edge * edge + !is_edge * detail;
    }
}

void interpolate_coefficients(const float** src_gaussian, const float** src_laplacian, float** dst_laplacian, const float ref,
                              const float ref_step, const int levelidx, const IM_SIZE_TYPE* level_sizes){
    const int current_size = level_sizes[levelidx].w * level_sizes[levelidx].h;
    const float ref_step_reciprocal = 1.0f / ref_step;
    int idx = 0;
#pragma omp parallel for
    for (idx = 0; idx < current_size; ++idx){
        float absdiff = (float)fabs(src_gaussian[levelidx][idx] - ref);
        dst_laplacian[levelidx][idx] = dst_laplacian[levelidx][idx] + (absdiff < ref_step) * src_laplacian[levelidx][idx] * (1 - absdiff * ref_step_reciprocal);
    }
}

void reconstruct_laplacian_pyramid(float** pyr, const int LLF_LEVEL, IM_SIZE_TYPE* level_sizes){
    int levelidx = 0;
    basic_algebra_ptr op = fadd; //add the upsample result
    for (levelidx = LLF_LEVEL - 1; levelidx > 0; --levelidx){
        filter_upsample(pyr[levelidx], pyr[levelidx - 1], op, level_sizes, levelidx - 1);
    }
}

void postprocessing(float* src, const double lum_max_clip, const double desired_exponent, const int numpixels){
    const double lum_max_clip_reciprocal = 1.0 / lum_max_clip;
    int idx = 0;
    for (idx = 0; idx < numpixels; ++idx){
        src[idx] = (float)pow(max(0.0, exp(src[idx]) * lum_max_clip_reciprocal), desired_exponent);
    }
}

void calc_hdr_bgr(const int num_exposure, const int overall_index, const unsigned char** exposure_ims, 
                  const float* log_exposure_time,  double* blue, double* green, double* red){
    //approximate log radiance from the smallest ev picture when the pixel is saturated in every pictures
    const int least_ev_index = num_exposure - 1;
    int exposures_idx = 0;
    float weighted_log_radiance_blue = 0.0f;
    float weighted_log_radiance_green = 0.0f;
    float weighted_log_radiance_red = 0.0f;
    int weight_sum_blue = 0;
    int weight_sum_green = 0;
    int weight_sum_red = 0;
    float radiance_temp = 0.0f;
    for (exposures_idx = 0; exposures_idx < num_exposure; ++exposures_idx){
        unsigned char zb = exposure_ims[exposures_idx][overall_index];
        unsigned char zg = exposure_ims[exposures_idx][overall_index + 1];
        unsigned char zr = exposure_ims[exposures_idx][overall_index + 2];
        if (zb == 0xFF || zg == 0xFF || zr == 0xFF){ //saturation protection
            weighted_log_radiance_blue = 0.0f;
            weighted_log_radiance_green = 0.0f;
            weighted_log_radiance_red = 0.0f;
            weight_sum_blue = 0;
            weight_sum_green = 0;
            weight_sum_red = 0;
        }
        else{
            weighted_log_radiance_blue += hdr_weights[zb] * (g_green[zb] - log_exposure_time[exposures_idx]);
            weighted_log_radiance_green += hdr_weights[zg] * (g_green[zg] - log_exposure_time[exposures_idx]);
            weighted_log_radiance_red += hdr_weights[zr] * (g_green[zr] - log_exposure_time[exposures_idx]);
            weight_sum_blue += hdr_weights[zb];
            weight_sum_green += hdr_weights[zg];
            weight_sum_red += hdr_weights[zr];
        }
    }
    //blue
    if (weight_sum_blue == 0){
        unsigned char pixel_value = exposure_ims[least_ev_index][overall_index];
        radiance_temp = g_green[pixel_value] - log_exposure_time[least_ev_index];
    }
    else{
        radiance_temp = weighted_log_radiance_blue / weight_sum_blue;
    }
    *blue = exp(radiance_temp);
    //green
    if (weight_sum_green == 0){
        unsigned char pixel_value = exposure_ims[least_ev_index][overall_index + 1];
        radiance_temp = g_green[pixel_value] - log_exposure_time[least_ev_index];
    }
    else{
        radiance_temp = weighted_log_radiance_green / weight_sum_green;
    }
    *green = exp(radiance_temp);
    //red
    if (weight_sum_red == 0){
        unsigned char pixel_value = exposure_ims[least_ev_index][overall_index + 2];
        radiance_temp = g_green[pixel_value] - log_exposure_time[least_ev_index];
    }
    else{
        radiance_temp = weighted_log_radiance_red / weight_sum_red;
    }
    *red = exp(radiance_temp);
}

void tone_mapping_local_laplacian(FLLF_INFO_TYPE* fllf_info){
    const int IMG_SIZE = IMG_WIDTH * IMG_HEIGHT;
    float* src = fllf_info->loglum;
    unsigned char* dst = fllf_info->ldr;
    const int LLF_LEVEL = (int)ceil(log(min(IMG_HEIGHT, IMG_WIDTH)) / log(2));
    //save pyramid size
    IM_SIZE_TYPE* level_sizes = (IM_SIZE_TYPE*)alloc_from_stack(LLF_LEVEL * sizeof(IM_SIZE_TYPE));
    float ref_step = (fllf_info->loglummax - fllf_info->loglummin) / (LLF_NUM_REF - 1);
    float ref = fllf_info->loglummin;
    int refidx = 0;
    //allocate space for result
    float** output_laplacian_pyr = alloc_empty_pyramid(LLF_LEVEL, level_sizes);
    //set stack anchor
    unsigned int stack_anchor = get_stack_current_alloc_size();
    //compute input Gaussian pyramid
    float** input_gaussian_pyr = build_gaussian_pyramid(src, LLF_LEVEL, level_sizes);
    float** remapped_loglum_laplacian_pyr = alloc_empty_pyramid(LLF_LEVEL, level_sizes);
    int levelidx = 0;
    for (refidx = 0; refidx < LLF_NUM_REF; ++refidx, ref += ref_step){
        remap_lum(src, remapped_loglum_laplacian_pyr[0], ref);
        build_laplacian_pyramid(remapped_loglum_laplacian_pyr, LLF_LEVEL, level_sizes);
        for (levelidx = 0; levelidx < LLF_LEVEL - 1; ++levelidx){
            interpolate_coefficients(input_gaussian_pyr, remapped_loglum_laplacian_pyr, output_laplacian_pyr, ref, ref_step, levelidx, level_sizes);
        }
    }
    //residual not affected
    memcpy(output_laplacian_pyr[LLF_LEVEL - 1], input_gaussian_pyr[LLF_LEVEL - 1], level_sizes[LLF_LEVEL - 1].w * level_sizes[LLF_LEVEL - 1].h * sizeof(float));
    //associate frequencies
    reconstruct_laplacian_pyramid(output_laplacian_pyr, LLF_LEVEL, level_sizes);
    //go back to previous stack anchor
    reset_stack_ptr_to_assigned_position(stack_anchor);
    float* outputlum = (float*)alloc_from_stack(IMG_SIZE * sizeof(float));
    memcpy(outputlum, output_laplacian_pyr[0], IMG_SIZE * sizeof(float));
    //postprocessing
#if 0
    float_sorting(output_laplacian_pyr[0], IMG_SIZE);
    const float LUM_CLIP = output_laplacian_pyr[0][(int)(IMG_SIZE * 0.995)] - output_laplacian_pyr[0][(int)(IMG_SIZE * 0.005)];
    const double LUM_MAX_CLIP = exp(output_laplacian_pyr[0][(int)(IMG_SIZE * 0.995)]);
    const double DESIRED_DR = 100.0;
    const double DESIRED_EXPONENT = log(DESIRED_DR) / LUM_CLIP;
#else
    float max_out = 0.0f;
    float min_out = 0.0f;
    float_max_min(output_laplacian_pyr[0], IMG_SIZE, &max_out, &min_out);
    const float LUM_CLIP = (max_out - min_out) * 0.99f;
    const double LUM_MAX_CLIP = exp(max_out - 0.05 * (max_out - min_out));
    const double DESIRED_DR = 100.0;
    const double DESIRED_EXPONENT = log(DESIRED_DR) / LUM_CLIP;
#endif 
    postprocessing(outputlum, LUM_MAX_CLIP, DESIRED_EXPONENT, IMG_SIZE);
    
    //color recovery
    int position_idx = 0;
    int overall_index = 0;
    for (position_idx = 0; position_idx < IMG_SIZE; ++position_idx, overall_index += 3){
        double blue = 0.0;
        double green = 0.0;
        double red = 0.0;
        calc_hdr_bgr(fllf_info->num_exposure, overall_index, fllf_info->exposure_images, fllf_info->exposure_values, &blue, &green, &red);
        double lum = (blue + 40.0 * green + 20.0 * red) * 0.0163934426;
        double outputratio = outputlum[position_idx] / lum;
        *dst = (unsigned char)(255.0 * pow(min(1.0, max(0.0, blue * outputratio)), LLF_GAMMA));
        ++dst;
        *dst = (unsigned char)(255.0 * pow(min(1.0, max(0.0, green * outputratio)), LLF_GAMMA));
        ++dst;
        *dst = (unsigned char)(255.0 * pow(min(1.0, max(0.0, red * outputratio)), LLF_GAMMA));
        ++dst;
    }
}

void build_hdr_image(FLLF_INFO_TYPE* fllf_info){
    const int IMG_SIZE = IMG_WIDTH * IMG_HEIGHT;
    int exposures_idx = 0;
    int position_idx = 0;
    int overall_index = 0;
    float* log_exposure_time = fllf_info->exposure_values;
    double max_log_lum = 0.0;
    double min_log_lum = DBL_MAX;
    for (exposures_idx = 0; exposures_idx < fllf_info->num_exposure; ++exposures_idx){
        fllf_info->exposure_values[exposures_idx] = (float)log(fllf_info->exposure_values[exposures_idx]);
    }    
    for (position_idx = 0; position_idx < IMG_SIZE; ++position_idx, overall_index += 3){
        double blue = 0.0;
        double green = 0.0;
        double red = 0.0;
        calc_hdr_bgr(fllf_info->num_exposure, overall_index, fllf_info->exposure_images, log_exposure_time, &blue, &green, &red);
        double log_lum = log(blue + 40.0 * green + 20.0 * red) - 4.11087386417; //to log domain for HDR input
        max_log_lum = max(log_lum, max_log_lum);
        min_log_lum = min(log_lum, min_log_lum);
        fllf_info->loglum[position_idx] = (float)log_lum; 
    }
    fllf_info->loglummax = (float)max_log_lum;
    fllf_info->loglummin = (float)min_log_lum;
}