from src.train_classificator import fit_by_indexes, fit_by_name
from src.models import get_castom_model_by_name
from src.metric import *

import torch
import os
import gc

# Установка типа девайса
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64

path_to_data = "../data/BraTS2020_training_data/content/data/"

Model_list = ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
              'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
              'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', #долго считается 'efficientnet_b6', #слишком долго считается 'efficientnet_b7',
              'efficientnet_v2_s', 'efficientnet_v2_m', #слишком долго считается  'efficientnet_v2_l',
              'densenet121', 'densenet161', 'densenet169', 'densenet201',
              'googlenet',
              'swin_t', 'swin_s', 'swin_b',
              'swin_v2_t', 'swin_v2_s', 'swin_v2_b'
              ]


use_sigmoid_list = [
                     True,
                     False
                    ]

learning_rate = 0.0001
num_epochs = 5



'''
5-10

number of all models 35
use device: cuda
use batch_size: 64
number of train: 45757, number of test 11438
        create model vgg11
        train model vgg11
        loss_function: MSELoss
        optimizer: Adam

sigmoid True

!RESULT! vgg11     __5 и 10_эпох__: [0.8813603520393372, 0.9003322124481201]
!RESULT! vgg11_bn  __5 и 10_эпох__: [0.9137961268424988, 0.9140583872795105]
!RESULT! vgg13     __5 и 10_эпох__: [0.9037418961524963, 0.9154572486877441]
!RESULT! vgg13_bn  __5 и 10_эпох__: [0.8881797194480896, 0.8958733677864075]
!RESULT! vgg16     __5 и 10_эпох__: [0.8974471092224121, 0.9001573324203491]
!RESULT! vgg16_bn  __5 и 10_эпох__: [0.886955738067627 , 0.8946493864059448]
!RESULT! vgg19     __5 и 10_эпох__: [0.892900824546814 , 0.9037418961524963]
!RESULT! vgg19_bn  __5 и 10_эпох__: [0.8748906850814819, 0.8996328115463257]

!RESULT! resnet18  __5 и 10_эпох__: [0.8925511240959167, 0.9015561938285828]
!RESULT! resnet34  __5 и 10_эпох__: [0.8730546832084656, 0.8823220729827881]
!RESULT! resnet50  __5 и 10_эпох__: [0.8354607224464417, 0.8511103391647339]
!RESULT! resnet101 __5 и 10_эпох__: [0.8718307018280029, 0.88022381067276]
!RESULT! resnet152 __5 и 10_эпох__: [0.8724427223205566, 0.8923762440681458]

!RESULT! efficientnet_b0 __5 и 10_эпох__: [0.889054000377655, 0.8923762440681458]
!RESULT! efficientnet_b1 __5 и 10_эпох__: [0.8803112506866455, 0.8936876654624939]
!RESULT! efficientnet_b2 __5 и 10_эпох__: [0.8588914275169373, 0.8793495297431946]
!RESULT! efficientnet_b3 __5 и 10_эпох__: [0.8635250926017761, 0.8848574757575989]
!RESULT! efficientnet_b4 __5 и 10_эпох__: [0.8385207056999207, 0.8567057251930237]
!RESULT! efficientnet_b5 __5 и 10_эпох__: [0.8512851595878601, 0.8690330386161804]
!RESULT! efficientnet_b6 __5 и 10_эпох__: [0.8616016507148743, 0.8755027055740356]

!RESULT! efficientnet_v2_s __5 и 10_эпох__: [0.8627382516860962, 0.8709564208984375]
!RESULT! efficientnet_v2_m __5 и 10_эпох__: [0.8030250072479248, 0.8203356862068176]

!RESULT! densenet121 __5 и 10_эпох__: [0.8964853882789612, 0.8928133845329285]
!RESULT! densenet161 __5 и 10_эпох__: [0.8936002850532532, 0.9062772989273071]
!RESULT! densenet169 __5 и 10_эпох__: [0.8928133845329285, 0.9088127017021179]
!RESULT! densenet201 __5 и 10_эпох__: [0.8929882645606995, 0.8984962105751038]

!RESULT! googlenet   __5 и 10_эпох__: [0.9030424952507019, 0.9123098254203796]

!RESULT! swin_t     __5 и 10_эпох__: [0.8299527764320374, 0.8604651093482971]
!RESULT! swin_s     __5 и 10_эпох__: [0.8547822833061218, 0.8665850758552551]
!RESULT! swin_b     __5 и 10_эпох__: [0.8872179985046387, 0.8981465101242065]
!RESULT! swin_v2_t  __5 и 10_эпох__: [0.8973596692085266, 0.9089875817298889]
!RESULT! swin_v2_s  __5 и 10_эпох__: [0.8916768431663513, 0.9014687538146973]
!RESULT! swin_v2_b  __5 и 10_эпох__: [0.8917642831802368, 0.9024304747581482]




sigmoid False

!RESULT! vgg11      __5 и 10_эпох__: [0.9024304747581482, 0.914145827293396]
!RESULT! vgg11_bn   __5 и 10_эпох__: [0.883283793926239 , 0.9023430347442627]
!RESULT! vgg13      __5 и 10_эпох__: [0.8796117901802063, 0.8778632283210754]
!RESULT! vgg13_bn   __5 и 10_эпох__: [0.8665850758552551, 0.8750655651092529]
!RESULT! vgg16      __5 и 10_эпох__: [0.8705192804336548, 0.8265430927276611]
!RESULT! vgg16_bn   __5 и 10_эпох__: [0.8291659355163574, 0.8605525493621826]
!RESULT! vgg19      __5 и 10_эпох__: [0.8931631445884705, 0.7382409572601318]
!RESULT! vgg19_bn   __5 и 10_эпох__: [0.8738415837287903, 0.8716558814048767]

!RESULT! resnet18  __5 и 10_эпох__: [0.8875677585601807, 0.9092498421669006]
!RESULT! resnet34  __5 и 10_эпох__: [0.8935128450393677, 0.9089001417160034]
!RESULT! resnet50  __5 и 10_эпох__: [0.8594159483909607, 0.8845952153205872]
!RESULT! resnet101 __5 и 10_эпох__: [0.856268584728241 , 0.8817100524902344]
!RESULT! resnet152 __5 и 10_эпох__: [0.861251950263977 , 0.8810106515884399]

!RESULT! efficientnet_b0 __5 и 10_эпох__: [0.8756775259971619, 0.8848574757575989]
!RESULT! efficientnet_b1 __5 и 10_эпох__: [0.8592411279678345, 0.8754152655601501]
!RESULT! efficientnet_b2 __5 и 10_эпох__: [0.8834586143493652, 0.8902779817581177]
!RESULT! efficientnet_b3 __5 и 10_эпох__: [0.8532960414886475, 0.871043860912323]
!RESULT! efficientnet_b4 __5 и 10_эпох__: [0.864311933517456, 0.8770763874053955]
!RESULT! efficientnet_b5 __5 и 10_эпох__: [0.8506731986999512, 0.8674593567848206]

!RESULT! efficientnet_v2_s __5 и 10_эпох__: [0.8506731986999512, 0.8593285083770752]
!RESULT! efficientnet_v2_m __5 и 10_эпох__: [0.8122923374176025, 0.8380835652351379]

# БЕЗ ВЫХОДНОЙ СИГМОИДЫ ДЕНСНЕТЫ ПОЛНОСТЬЮ ПЕРЕТРЕНИРОВЫВАЮТСЯ (первая эпоха 0.27)
!RESULT! densenet121 __5 и 10_эпох__: [0.6313166618347168, 0.729585587978363]
!RESULT! densenet161 __5 и 10_эпох__: [0.5721279978752136, 0.2426997721195221]
!RESULT! densenet169 __5 и 10_эпох__: [0.518971860408783,  0.6403217315673828]
!RESULT! densenet201 __5 и 10_эпох__: [0.6074488162994385, 0.6791396737098694]

!RESULT! googlenet __5 и 10_эпох__: [0.9084630012512207, 0.9150201082229614]

!RESULT! swin_t __5 и 10_эпох__: [0.8435040712356567, 0.8775135278701782]
!RESULT! swin_s __5 и 10_эпох__: [0.849973738193512 , 0.9002447724342346]
!RESULT! swin_b __5 и 10_эпох__: [0.8891414403915405, 0.8976219296455383]

!RESULT! swin_v2_t __5 и 10_эпох__: [0.889840841293335, 0.9061024188995361]
!RESULT! swin_v2_s __5 и 10_эпох__: [0.8636125326156616, 0.894999086856842]
!RESULT! swin_v2_b __5 и 10_эпох__: [0.8865185976028442, 0.9006819128990173]


[('use_sigmoid_fun: True', [('googlenet', [0.894999086856842, 0.9205280542373657])]),
 ('use_sigmoid_fun: False', [('googlenet', [0.904965877532959, 0.9125720858573914])])]

'''


"""
15-20

sigmoid True

!RESULT! vgg11      __15 и 20_эпох__: [0.9235005974769592, 0.9255988597869873]
!RESULT! vgg11_bn   __15 и 20_эпох__: [0.9396747350692749, 0.9194788932800293]
!RESULT! vgg13      __15 и 20_эпох__: [0.9332925081253052, 0.932767927646637]
!RESULT! vgg13_bn   __15 и 20_эпох__: [0.9332050681114197, 0.9210526347160339]
!RESULT! vgg16      __15 и 20_эпох__: [0.9214023351669312, 0.92603600025177]
!RESULT! vgg16_bn   __15 и 20_эпох__: [0.9256862998008728, 0.919916033744812]
!RESULT! vgg19      __15 и 20_эпох__: [0.9318936467170715, 0.9311942458152771]
!RESULT! vgg19_bn   __15 и 20_эпох__: [0.9267354011535645, 0.9219269156455994]


!RESULT! resnet18 __15 и 20_эпох__: [0.9158943891525269, 0.9061898589134216]
!RESULT! resnet34 __15 и 20_эпох__: [0.895173966884613 , 0.8992830514907837]
!RESULT! resnet50 __15 и 20_эпох__: [0.8804860711097717, 0.895173966884613]
!RESULT! resnet101 __5 и 10_эпох__: [0.895348846912384 , 0.898059070110321]
!RESULT! resnet152 __5 и 10_эпох__: [0.8907151222229004, 0.8962231278419495]

!RESULT! efficientnet_b0 __5 и 10_эпох__: [0.9011190533638   , 0.9033921957015991]
!RESULT! efficientnet_b1 __5 и 10_эпох__: [0.9028676152229309, 0.8994579315185547]
!RESULT! efficientnet_b2 __5 и 10_эпох__: [0.874278724193573 , 0.8842454552650452]
!RESULT! efficientnet_b3 __5 и 10_эпох__: [0.9017310738563538, 0.9008567929267883]
!RESULT! efficientnet_b4 __5 и 10_эпох__: [0.8774260878562927, 0.8809232115745544]
!RESULT! efficientnet_b5 __5 и 10_эпох__: [0.8924636840820312, 0.8942996859550476]

!RESULT! efficientnet_v2_s __5 и 10_эпох__: [0.8748032450675964, 0.8870431780815125]
!RESULT! efficientnet_v2_m __5 и 10_эпох__: [0.8205980062484741, 0.8270676732063293]

!RESULT! densenet121 __5 и 10_эпох__: [0.904965877532959 , 0.9084630012512207]
!RESULT! densenet161 __5 и 10_эпох__: [0.9224514365196228, 0.9153698086738586]
!RESULT! densenet169 __5 и 10_эпох__: [0.9201783537864685, 0.9205280542373657]
!RESULT! densenet201 __5 и 10_эпох__: [0.9135338068008423, 0.9092498421669006]

!RESULT! googlenet __15 и 20_эпох__: [0.9191291928291321, 0.9234131574630737]

!RESULT! swin_t __15 и 20_эпох__: [0.8833711743354797, 0.873928964138031]
!RESULT! swin_s __15 и 20_эпох__: [0.8831963539123535, 0.8910648822784424]
!RESULT! swin_b __15 и 20_эпох__: [0.9132715463638306, 0.913970947265625]

!RESULT! swin_v2_t __15 и 20_эпох__: [0.9161566495895386, 0.9164189100265503]
!RESULT! swin_v2_s __15 и 20_эпох__: [0.9142332673072815, 0.9202657341957092]
!RESULT! swin_v2_b __15 и 20_эпох__: [0.9146704077720642, 0.9146704077720642]

sigmoid False

!RESULT! vgg11      __15 и 20_эпох__: [0.9187794923782349, 0.9293582439422607]
!RESULT! vgg11_bn   __15 и 20_эпох__: [0.9287462830543518, 0.9197412133216858]
!RESULT! vgg13      __15 и 20_эпох__: [0.9060150384902954, 0.9187794923782349]
!RESULT! vgg13_bn   __15 и 20_эпох__: [0.9207903146743774, 0.9179926514625549]
!RESULT! vgg16      __15 и 20_эпох__: [0.8778632283210754, 0.9033047556877136]
!RESULT! vgg16_bn   __15 и 20_эпох__: [0.9040041565895081, 0.8987585306167603]
!RESULT! vgg19      __15 и 20_эпох__: [0.9104738235473633, 0.9130092263221741]
!RESULT! vgg19_bn   __15 и 20_эпох__: [0.898845911026001 , 0.8952614068984985]

!RESULT! resnet18  __15 и 20_эпох__: [0.9203531742095947, 0.9194788932800293]
!RESULT! resnet34  __15 и 20_эпох__: [0.9211400151252747, 0.9130966663360596]
!RESULT! resnet50  __15 и 20_эпох__: [0.8910648822784424, 0.8781255483627319]
!RESULT! resnet101 __15 и 20_эпох__: [0.8894911408424377, 0.8917642831802368]
!RESULT! resnet152 __15 и 20_эпох__: [0.8854694962501526, 0.8812729120254517]

!RESULT! efficientnet_b0 __15 и 20_эпох__: [0.8922014236450195, 0.8987585306167603]
!RESULT! efficientnet_b1 __15 и 20_эпох__: [0.8909774422645569, 0.8916768431663513]
!RESULT! efficientnet_b2 __15 и 20_эпох__: [0.8933379650115967, 0.8993704915046692]
!RESULT! efficientnet_b3 __15 и 20_эпох__: [0.8737541437149048, 0.8786500692367554]
!RESULT! efficientnet_b4 __15 и 20_эпох__: [0.8772512674331665, 0.8796117901802063]
!RESULT! efficientnet_b5 __15 и 20_эпох__: [0.8945619463920593, 0.890190601348877]

!RESULT! efficientnet_v2_s __15 и 20_эпох__: [0.855918824672699 , 0.8736667037010193]
!RESULT! efficientnet_v2_m __15 и 20_эпох__: [0.8331875801086426, 0.8626508116722107]

!RESULT! densenet121 __15 и 20_эпох__: [0.7196187973022461, 0.6731945872306824]
!RESULT! densenet161 __15 и 20_эпох__: [0.5966952443122864, 0.34184297919273376]
!RESULT! densenet169 __15 и 20_эпох__: [0.5794719457626343, 0.7255638837814331]
!RESULT! densenet201 __15 и 20_эпох__: [0.6968875527381897, 0.7012589573860168]

!RESULT! googlenet __15 и 20_эпох__: [0.9210526347160339, 0.8899282813072205]

!RESULT! swin_t __15 и 20_эпох__: [0.908025860786438 , 0.9034796357154846]
!RESULT! swin_s __15 и 20_эпох__: [0.9133589863777161, 0.9165063500404358]
!RESULT! swin_b __15 и 20_эпох__: [0.8964853882789612, 0.8822346329689026]

!RESULT! swin_v2_t __15 и 20_эпох__: [0.910910964012146 , 0.9031298756599426]
!RESULT! swin_v2_s __15 и 20_эпох__: [0.9176429510116577, 0.9261234402656555]
!RESULT! swin_v2_b __15 и 20_эпох__: [0.9214023351669312, 0.9175555109977722]

"""


"""
25-30 без заморозки весов

sigmoid True

!RESULT! vgg11      __25 и 30_эпох__: [0.938275933265686 , 0.9388004541397095]
!RESULT! vgg11_bn   __25 и 30_эпох__: [0.9401993155479431, 0.9360902309417725]
!RESULT! vgg13      __25 и 30_эпох__: [0.9385381937026978, 0.9318062663078308]
!RESULT! vgg13_bn   __25 и 30_эпох__: [0.9322434067726135, 0.938887894153595]
!RESULT! vgg16      __25 и 30_эпох__: [0.9422975778579712, 0.938275933265686]
!RESULT! vgg16_bn   __25 и 30_эпох__: [0.9460569620132446, 0.9272599816322327]
!RESULT! vgg19      __25 и 30_эпох__: [0.9415107369422913, 0.947892963886261]
!RESULT! vgg19_bn   __25 и 30_эпох__: [0.9396747350692749, 0.9426472783088684]

!RESULT! resnet18  __25 и 30_эпох__: [0.9459695816040039, 0.9347788095474243]
!RESULT! resnet34  __25 и 30_эпох__: [0.936002790927887 , 0.9206154942512512]
!RESULT! resnet50  __25 и 30_эпох__: [0.9408113360404968, 0.9213148951530457]
!RESULT! resnet101 __25 и 30_эпох__: [0.9335548281669617, 0.9392375946044922]
!RESULT! resnet152 __25 и 30_эпох__: [0.9397621750831604, 0.9433467388153076]

!RESULT! efficientnet_b0 __25 и 30_эпох__: [0.9386256337165833, 0.9422975778579712]
!RESULT! efficientnet_b1 __25 и 30_эпох__: [0.944046139717102 , 0.9449204206466675]
!RESULT! efficientnet_b2 __25 и 30_эпох__: [0.9370518922805786, 0.9332925081253052]
!RESULT! efficientnet_b3 __25 и 30_эпох__: [0.9446581602096558, 0.9422101378440857]
!RESULT! efficientnet_b4 __25 и 30_эпох__: [0.9408113360404968, 0.9381884932518005]
!RESULT! efficientnet_b5 __25 и 30_эпох__: [0.9449204206466675, 0.9282217025756836]

!RESULT! efficientnet_v2_s __25 и 30_эпох__: [0.9436964392662048, 0.9413358569145203]
!RESULT! efficientnet_v2_m __25 и 30_эпох__: [0.941947877407074 , 0.9399370551109314]

!RESULT! densenet121 __25 и 30_эпох__: [0.9469312429428101, 0.9433467388153076]
!RESULT! densenet161 __25 и 30_эпох__: [0.9182549118995667, 0.9462318420410156]
!RESULT! densenet169 __25 и 30_эпох__: [0.938713014125824 , 0.9407238960266113]
!RESULT! densenet201 __25 и 30_эпох__: [0.9228885769844055, 0.9255114197731018]

!RESULT! googlenet __25 и 30_эпох__: [0.9349536299705505, 0.932155966758728]

!RESULT! swin_t __25 и 30_эпох__: [0.9388004541397095, 0.935041069984436]
!RESULT! swin_s __25 и 30_эпох__: [0.947281002998352 , 0.9431718587875366]
!RESULT! swin_b __25 и 30_эпох__: [0.9473683834075928, 0.9489421248435974]

!RESULT! swin_v2_t __25 и 30_эпох__: [0.947281002998352 , 0.938275933265686]
!RESULT! swin_v2_s __25 и 30_эпох__: [0.9460569620132446, 0.9499912261962891]
!RESULT! swin_v2_b __25 и 30_эпох__: [0.9426472783088684, 0.9491169452667236]


sigmoid False

!RESULT! vgg11      __25 и 30_эпох__: [0.9420353174209595, 0.936002790927887]
!RESULT! vgg11_bn   __25 и 30_эпох__: [0.9354782104492188, 0.92682284116745]
!RESULT! vgg13      __25 и 30_эпох__: [0.9381884932518005, 0.938101053237915]
!RESULT! vgg13_bn   __25 и 30_эпох__: [0.9322434067726135, 0.9364399313926697]
!RESULT! vgg16      __25 и 30_эпох__: [0.941161036491394 , 0.9398496150970459]
!RESULT! vgg16_bn   __25 и 30_эпох__: [0.9455324411392212, 0.938275933265686]
!RESULT! vgg19      __25 и 30_эпох__: [0.9485923647880554, 0.9431718587875366]
!RESULT! vgg19_bn   __25 и 30_эпох__: [0.9433467388153076, 0.9450953006744385]

!RESULT! resnet18 __25 и 30_эпох__: [0.9441335797309875, 0.9467564225196838]
!RESULT! resnet34 __25 и 30_эпох__: [0.9420353174209595, 0.950778067111969]
!RESULT! resnet50 __25 и 30_эпох__: [0.9444832801818848, 0.9424724578857422]
!RESULT! resnet101 __25 и 30_эпох__:[0.9412484765052795, 0.9465815424919128]
!RESULT! resnet152 __25 и 30_эпох__:[0.9371393322944641, 0.934866189956665]

!RESULT! efficientnet_b0 __25 и 30_эпох__: [0.9415981769561768, 0.9429970383644104]
!RESULT! efficientnet_b1 __25 и 30_эпох__: [0.922976016998291 , 0.9391501545906067]
!RESULT! efficientnet_b2 __25 и 30_эпох__: [0.9377513527870178, 0.9430844187736511]
!RESULT! efficientnet_b3 __25 и 30_эпох__: [0.9473683834075928, 0.9460569620132446]
!RESULT! efficientnet_b4 __25 и 30_эпох__: [0.9413358569145203, 0.9449204206466675]
!RESULT! efficientnet_b5 __25 и 30_эпох__: [0.9212274551391602, 0.9424724578857422]

!RESULT! efficientnet_v2_s __25 и 30_эпох__: [0.9394124746322632, 0.9325931072235107]
!RESULT! efficientnet_v2_m __25 и 30_эпох__: [0.9405490159988403, 0.932942807674408]

!RESULT! densenet121 __25 и 30_эпох__: [0.9408113360404968, 0.9434341192245483]
!RESULT! densenet161 __25 и 30_эпох__: [0.8884420394897461, 0.9502535462379456]
!RESULT! densenet169 __25 и 30_эпох__: [0.9274348616600037, 0.9205280542373657]
!RESULT! densenet201 __25 и 30_эпох__: [0.9117852449417114, 0.9453575611114502]

!RESULT! googlenet __25 и 30_эпох__: [0.9425598978996277, 0.9429095983505249]

!RESULT! swin_t __25 и 30_эпох__: [0.9447455406188965, 0.9365273714065552]
!RESULT! swin_s __25 и 30_эпох__: [0.9462318420410156, 0.9433467388153076]
!RESULT! swin_b __25 и 30_эпох__: [0.9406364560127258, 0.947281002998352]
 
!RESULT! swin_v2_t __25 и 30_эпох__: [0.9496415257453918, 0.9523518085479736]
!RESULT! swin_v2_s __25 и 30_эпох__: [0.9485923647880554, 0.9518272280693054]
!RESULT! swin_v2_b __25 и 30_эпох__: [0.9428221583366394, 0.9436964392662048]
"""

"""
35-40 без заморозки весов

sigmoid True

!RESULT! vgg11      __35 и 40_эпох__: [0.9365273714065552, 0.935215950012207]
!RESULT! vgg11_bn   __35 и 40_эпох__: [0.9346913695335388, 0.9430844187736511]
!RESULT! vgg13      __35 и 40_эпох__: [0.934866189956665 , 0.9248994588851929]
!RESULT! vgg13_bn   __35 и 40_эпох__: [0.9355656504631042, 0.9308445453643799]
!RESULT! vgg16      __35 и 40_эпох__: [0.9399370551109314, 0.9359153509140015]
!RESULT! vgg16_bn   __35 и 40_эпох__: [0.9307571053504944, 0.9389753341674805]
!RESULT! vgg19      __35 и 40_эпох__: [0.9439586997032166, 0.942122757434845]
!RESULT! vgg19_bn   __35 и 40_эпох__: [0.9363524913787842, 0.947281002998352]

!RESULT! resnet18  __35 и 40_эпох__: [0.9434341192245483, 0.9394999146461487]
!RESULT! resnet34  __35 и 40_эпох__: [0.9322434067726135, 0.9464067220687866]
!RESULT! resnet50  __35 и 40_эпох__: [0.9339045286178589, 0.9354782104492188]
!RESULT! resnet101 __35 и 40_эпох__: [0.9373142123222351, 0.9443958401679993]
!RESULT! resnet152 __35 и 40_эпох__: [0.9469312429428101, 0.9356530904769897]

!RESULT! efficientnet_b0 __35 и 40_эпох__: [0.9372267723083496, 0.9415981769561768]
!RESULT! efficientnet_b1 __35 и 40_эпох__: [0.9366147518157959, 0.9410735964775085]
!RESULT! efficientnet_b2 __35 и 40_эпох__: [0.9337296485900879, 0.933117687702179]
!RESULT! efficientnet_b3 __35 и 40_эпох__: [0.9248120188713074, 0.9362650513648987]
!RESULT! efficientnet_b4 __35 и 40_эпох__: [0.9318062663078308, 0.9401993155479431]
!RESULT! efficientnet_b5 __35 и 40_эпох__: [0.9402867555618286, 0.9446581602096558]

!RESULT! efficientnet_v2_s __35 и 40_эпох__: [0.935215950012207 , 0.940986156463623]
!RESULT! efficientnet_v2_m __35 и 40_эпох__: [0.9369645118713379, 0.9345164895057678]

!RESULT! densenet121 __35 и 40_эпох__: [0.938713014125824 , 0.9374016523361206]
!RESULT! densenet161 __35 и 40_эпох__: [0.9512152075767517, 0.947281002998352]
!RESULT! densenet169 __35 и 40_эпох__: [0.947106122970581 , 0.9483301043510437]
!RESULT! densenet201 __35 и 40_эпох__: [0.9457947015762329, 0.9436964392662048]

!RESULT! googlenet __35 и 40_эпох__: [0.9391501545906067, 0.932155966758728]

!RESULT! swin_t __35 и 40_эпох__: [0.9413358569145203, 0.9431718587875366]
!RESULT! swin_s __35 и 40_эпох__: [0.9339045286178589, 0.9473683834075928]
!RESULT! swin_b __35 и 40_эпох__: [0.944832980632782 , 0.9404615759849548]

!RESULT! swin_v2_t __35 и 40_эпох__: [0.945007860660553, 0.9496415257453918]
!RESULT! swin_v2_s __35 и 40_эпох__: [0.9467564225196838, 0.9485923647880554]
!RESULT! swin_v2_b __35 и 40_эпох__: [0.9408987164497375, 0.9447455406188965]


sigmoid False


!RESULT! vgg11      __35 и 40_эпох__: [0.9431718587875366, 0.9320685267448425]
!RESULT! vgg11_bn   __35 и 40_эпох__: [0.9245496988296509, 0.9312816858291626]
!RESULT! vgg13      __35 и 40_эпох__: [0.9465815424919128, 0.9374890327453613]
!RESULT! vgg13_bn   __35 и 40_эпох__: [0.935041069984436 , 0.939062774181366]
!RESULT! vgg16      __35 и 40_эпох__: [0.9453575611114502, 0.936002790927887]
!RESULT! vgg16_bn   __35 и 40_эпох__: [0.942122757434845 , 0.9433467388153076]
!RESULT! vgg19      __35 и 40_эпох__: [0.9431718587875366, 0.9455324411392212]
!RESULT! vgg19_bn   __35 и 40_эпох__: [0.938101053237915 , 0.9301450848579407]

!RESULT! resnet18  __35 и 40_эпох__: [0.941772997379303 , 0.9340793490409851]
!RESULT! resnet34  __35 и 40_эпох__: [0.9369645118713379, 0.9477181434631348]
!RESULT! resnet50  __35 и 40_эпох__: [0.9315439462661743, 0.9326805472373962]
!RESULT! resnet101 __35 и 40_эпох__: [0.938101053237915, 0.9477181434631348]
!RESULT! resnet152 __35 и 40_эпох__: [0.9290085434913635, 0.9349536299705505]

!RESULT! efficientnet_b0 __35 и 40_эпох__: [0.9360902309417725, 0.9423850178718567]
!RESULT! efficientnet_b1 __35 и 40_эпох__: [0.935215950012207 , 0.9388004541397095]
!RESULT! efficientnet_b2 __35 и 40_эпох__: [0.944046139717102 , 0.940986156463623]
!RESULT! efficientnet_b3 __35 и 40_эпох__: [0.9338170886039734, 0.9392375946044922]
!RESULT! efficientnet_b4 __35 и 40_эпох__: [0.9365273714065552, 0.9381884932518005]
!RESULT! efficientnet_b5 __35 и 40_эпох__: [0.9336422085762024, 0.9322434067726135]

!RESULT! efficientnet_v2_s __35 и 40_эпох__: [0.9351285099983215, 0.9336422085762024]
!RESULT! efficientnet_v2_m __35 и 40_эпох__: [0.9375764727592468, 0.9391501545906067]

!RESULT! densenet121 __35 и 40_эпох__: [0.9463192820549011, 0.9435215592384338]
!RESULT! densenet161 __35 и 40_эпох__: [0.9456198215484619, 0.9490295052528381]
!RESULT! densenet169 __35 и 40_эпох__: [0.9393250346183777, 0.9180800914764404]
!RESULT! densenet201 __35 и 40_эпох__: [0.9344290494918823, 0.9450953006744385]

!RESULT! googlenet __35 и 40_эпох__: [0.9359153509140015, 0.9412484765052795]

!RESULT! swin_t __35 и 40_эпох__: [0.9450953006744385, 0.9501661062240601]
!RESULT! swin_s __35 и 40_эпох__: [0.9424724578857422, 0.9453575611114502]
!RESULT! swin_b __35 и 40_эпох__: [0.9333799481391907, 0.9400244355201721]

!RESULT! swin_v2_t __35 и 40_эпох__: [0.9456198215484619, 0.9485923647880554]
!RESULT! swin_v2_s __35 и 40_эпох__: [0.9359153509140015, 0.9433467388153076]
!RESULT! swin_v2_b __35 и 40_эпох__: [0.9475432634353638, 0.9485049843788147]

"""

              
print("number of all models", len(Model_list))


def get_train_test_from_list(list_of_name, prop_of_train=0.8):
    # перемешать случайно список
    #np.random.shuffle(list_of_name)
    
    # взять первые 80% на трейн, и 20 на тест
    len_of_train = int(round(prop_of_train*len(list_of_name)))+1
    return list_of_name[:len_of_train], list_of_name[len_of_train:]

def extract_numbers_from_path(path):
    # Получаем имя файла
    filename = os.path.basename(path)
    # Удаляем расширение файла
    filename_without_ext = os.path.splitext(filename)[0]
    # Разделяем по подчеркиваниям
    parts = filename_without_ext.split('_')
    # parts[1] - число volume, parts[3] - число slice
    volume_num = int(parts[1])
    slice_num = int(parts[3])
    return volume_num, slice_num

list_of_data_names = [os.path.join(path_to_data, name_data) for name_data in os.listdir(path_to_data) if name_data.endswith(".h5")]
#list_of_data_names = list_of_data_names[:121]

# Сортировка по извлеченным числам
list_of_data_names = sorted(list_of_data_names, key=extract_numbers_from_path)
indexes = [i for i in range(len(list_of_data_names))]


train_indexes, test_indexes = get_train_test_from_list(indexes)

train_names = [list_of_data_names[i] for i in train_indexes]
test_names = [list_of_data_names[i] for i in test_indexes]

#print(test_names)

# не влазивает в память
#slises, marks = read_dataset(list_of_data_names)
#dataset = dataset_to_torch(slises, marks, device)

print(f"use device: {device}")
print(f"use batch_size: {batch_size}")
print(f"number of train: {len(train_indexes)}, number of test {len(test_indexes)}")


'''
# Обучение после инициализации весов
all_res = []
for use_sigmoid_fun in use_sigmoid_list:
    model_res = []
    for model_name in Model_list:
        # при запуске нескольких экспериментов забивается память
        with torch.no_grad():
            torch.cuda.empty_cache()
        gc.collect()

        print(f"\tcreate model {model_name}")

        class_model = get_castom_model_by_name(model_name)
        model = class_model((240, 240, 4), True, model_name, 1, use_sigmoid_fun)
        model.block_requires_grad_features()
        model.to(device)

        print(f"\ttrain model {model_name}")

        loss_function = torch.nn.MSELoss()
        optimizer_exp = torch.optim.Adam(model.parameters(), lr = learning_rate)
        print("\tloss_function: MSELoss")
        print("\toptimizer: Adam")

        model_test_res = []
        for i in range(2):
            print(f"\t\tStart training {5*(i+1)} epoch")
            trained_model = fit_by_name(model, train_names, num_epochs, optimizer_exp, loss_function, batch_size, device)
            print('\t\tFinished Training')
            # сохранить только веса
            torch.save(trained_model.state_dict(), f"exp_all/model_{model_name}{'_sigmoid' if use_sigmoid_fun else ''}" + '.pth')

            test_res_accuracy = get_accuracy(tqdm(dataset_gen(test_names, batch_size, device)), trained_model, device) #################################################### test_names train_names
            print(f"\t\tTest result {test_res_accuracy.item()}")
            model_test_res.append(test_res_accuracy.item())

        model_res.append((model_name, model_test_res))
        print("!RESULT!", model_name, "__5 и 10_эпох__:", model_test_res)
    all_res.append((f"use_sigmoid_fun: {use_sigmoid_fun}", model_res))
print(all_res)
'''

'''
# Продолжение обучения
all_res = []
for use_sigmoid_fun in use_sigmoid_list:
    print("use_sigmoid_fun", use_sigmoid_fun)
    model_res = []
    for model_name in Model_list:
        # при запуске нескольких экспериментов забивается память
        with torch.no_grad():
            torch.cuda.empty_cache()
        gc.collect()

        print(f"\tcreate model {model_name}")

        class_model = get_castom_model_by_name(model_name)
        model = class_model((240, 240, 4), False, model_name, 1, use_sigmoid_fun)
        model.load_state_dict(torch.load(f"exp_all/model_{model_name}{'_sigmoid' if use_sigmoid_fun else ''}" + '.pth', weights_only=True))
        model.block_requires_grad_features()
        model.to(device)

        print(f"\ttrain model {model_name}")

        loss_function = torch.nn.MSELoss()
        optimizer_exp = torch.optim.Adam(model.parameters(), lr = learning_rate)
        print("\tloss_function: MSELoss")
        print("\toptimizer: Adam")

        model_test_res = []
        for i in range(2):
            print(f"\t\tStart training {5*(i+1)+10} epoch")
            trained_model = fit_by_name(model, train_names, num_epochs, optimizer_exp, loss_function, batch_size, device)
            print('\t\tFinished Training')
            # сохранить только веса
            torch.save(trained_model.state_dict(), f"exp_all_v2/model_{model_name}{'_sigmoid' if use_sigmoid_fun else ''}" + '.pth')

            test_res_accuracy = get_accuracy(tqdm(dataset_gen(test_names, batch_size, device)), trained_model, device) #################################################### test_names train_names
            print(f"\t\tTest result {test_res_accuracy.item()}")
            model_test_res.append(test_res_accuracy.item())

        model_res.append((model_name, model_test_res))
        print("!RESULT!", model_name, "__15 и 20_эпох__:", model_test_res)
    all_res.append((f"use_sigmoid_fun: {use_sigmoid_fun}", model_res))
print(all_res)
'''

'''
# Продолжение обучения без заморозки весов
all_res = []
for use_sigmoid_fun in use_sigmoid_list:
    print("use_sigmoid_fun", use_sigmoid_fun)
    model_res = []
    for model_name in Model_list:
        # при запуске нескольких экспериментов забивается память
        with torch.no_grad():
            torch.cuda.empty_cache()
        gc.collect()

        print(f"\tcreate model {model_name}")

        class_model = get_castom_model_by_name(model_name)
        model = class_model((240, 240, 4), False, model_name, 1, use_sigmoid_fun)
        model.load_state_dict(torch.load(f"exp_all_v2/model_{model_name}{'_sigmoid' if use_sigmoid_fun else ''}" + '.pth', weights_only=True))
        #model.block_requires_grad_features()
        model.to(device)

        print(f"\ttrain model {model_name}")

        loss_function = torch.nn.MSELoss()
        optimizer_exp = torch.optim.Adam(model.parameters(), lr = learning_rate)
        print("\tloss_function: MSELoss")
        print("\toptimizer: Adam")

        model_test_res = []
        for i in range(2):
            print(f"\t\tStart training {5*(i+1)+20} epoch")
            trained_model = fit_by_name(model, train_names, num_epochs, optimizer_exp, loss_function, batch_size, device)
            print('\t\tFinished Training')
            # сохранить только веса
            torch.save(trained_model.state_dict(), f"exp_all_v3/model_{model_name}{'_sigmoid' if use_sigmoid_fun else ''}" + '.pth')

            test_res_accuracy = get_accuracy(tqdm(dataset_gen(test_names, batch_size, device)), trained_model, device) #################################################### test_names train_names
            print(f"\t\tTest result {test_res_accuracy.item()}")
            model_test_res.append(test_res_accuracy.item())

        model_res.append((model_name, model_test_res))
        print("!RESULT!", model_name, "__25 и 30_эпох__:", model_test_res)
    all_res.append((f"use_sigmoid_fun: {use_sigmoid_fun}", model_res))
print(all_res)
'''

# Продолжение обучения 2 без заморозки весов
all_res = []
for use_sigmoid_fun in use_sigmoid_list:
    print("use_sigmoid_fun", use_sigmoid_fun)
    model_res = []
    for model_name in Model_list:
        # при запуске нескольких экспериментов забивается память
        with torch.no_grad():
            torch.cuda.empty_cache()
        gc.collect()

        print(f"\tcreate model {model_name}")

        class_model = get_castom_model_by_name(model_name)
        model = class_model((240, 240, 4), False, model_name, 1, use_sigmoid_fun)
        model.load_state_dict(torch.load(f"exp_all_v3/model_{model_name}{'_sigmoid' if use_sigmoid_fun else ''}" + '.pth', weights_only=True))
        #model.block_requires_grad_features()
        model.to(device)

        print(f"\ttrain model {model_name}")

        loss_function = torch.nn.MSELoss()
        optimizer_exp = torch.optim.Adam(model.parameters(), lr = learning_rate)
        print("\tloss_function: MSELoss")
        print("\toptimizer: Adam")

        model_test_res = []
        for i in range(2):
            print(f"\t\tStart training {5*(i+1)+30} epoch")
            trained_model = fit_by_name(model, train_names, num_epochs, optimizer_exp, loss_function, batch_size, device)
            print('\t\tFinished Training')
            # сохранить только веса
            torch.save(trained_model.state_dict(), f"exp_all_v4/model_{model_name}{'_sigmoid' if use_sigmoid_fun else ''}" + '.pth')

            test_res_accuracy = get_accuracy(tqdm(dataset_gen(test_names, batch_size, device)), trained_model, device) #################################################### test_names train_names
            print(f"\t\tTest result {test_res_accuracy.item()}")
            model_test_res.append(test_res_accuracy.item())

        model_res.append((model_name, model_test_res))
        print("!RESULT!", model_name, "__35 и 40_эпох__:", model_test_res)
    all_res.append((f"use_sigmoid_fun: {use_sigmoid_fun}", model_res))
print(all_res)
