import gradio as gr
import cv2

from src.comparison import CalulateMetricsFromModelPredict
from src.prepare_data import plug_old_connect_res_test_data
from src.test import getPipliner, test_data
from src.prepare_data import to_0_1_format_img
import os

etal_path="test_data/img_etal"


def read_image(image_path, device="cpu"):
    img_for_test = cv2.imread(image_path, 0)
    dataset_for_predict = [([to_0_1_format_img(img_for_test)], [os.path.basename(image_path)])]
    return img_for_test, dataset_for_predict

def init_model():
    model, name_model = getPipliner("test_data/model_data/",
                                           "config_diffusion_data_42_slices_6_classes_dataset_mix_6_classes_seed_1466947709_tiny_unet_v3",
                                           device='cpu')
    return model, name_model                           

def handle_file(file_name):
    img, data = read_image(file_name)
    # После выбора файла скрыть input и показать изображение
    return gr.update(visible=False), gr.update(visible=True, value=img), data

def reset():
    # Обнуляем значение file_input и показываем его
    return gr.update(value=None, visible=True), gr.update(value=None, visible=False), None

# Обработка при нажатии кнопки
def segmentate_data(dataset_for_predict):
    model, name_model = init_model()
    tiling_mode = True
    tiled_data = {"size": 256, "overlap": 128, "unique_area": 0} if tiling_mode else None
    class_names=[
            "mitochondria",
            "PSD",
            "vesicles",
            "axon",
            "boundaries",
            "mitochondrial boundaries"
            ]

    print(name_model)
    print(model.device)

    predict_img_list, predict_name_list = test_data(model,
                                                    dataset_for_predict,
                                                    save_mask_dir=None,
                                                    tiled_data=tiled_data,
                                                    batch_size=1
                                                    )
    
    predict_for_check = plug_old_connect_res_test_data(predict_img_list, predict_name_list)
    
    result_metrics_merge,\
    text_result_merge,\
    text_result_merge_all = CalulateMetricsFromModelPredict(predict_for_check,
                                                            name_model,
                                                            6,
                                                            etal_path=etal_path,
                                                            class_names=class_names,
                                                            using_metric_names=["Dice", "Jaccard"],
                                                            merge_images=True,
                                                            is_print_metric=False
                                                            )
        
    masks_label_list = []
    for i, img in enumerate(predict_img_list):
        for y in range(2):
            for x in range(3):
                c_i = y*3+x
                classname = class_names[c_i]
                masks_label_list.append((img[:,:,c_i], classname))
    
    str_dict = ""
    for key, value in result_metrics_merge.items():
        if key == 'Model_name':
            str_dict += f'{key}: {value}\n'
        elif key == 'Metrics': 
            str_dict += f'{key}\n'
            
            max_char_len = max([len(key) for key in value[0].keys()])
            
            for class_name, val in value[0].items():
                temp_str = [f"'{metric_name}': {val_met:.3f}" for metric_name, val_met in val.items()]
                str_dict += f'\t{class_name+"_"*(max_char_len-len(class_name))}: {", ".join(temp_str)}\n'
    
    
    
    return gr.update(value=str_dict), gr.update(value=masks_label_list, label=predict_name_list[0])


with gr.Blocks(fill_height=True) as demo:
    data = gr.State()

    gr.Markdown("Выберите изображение, задайте параметры и посмотрите результат.")
    
    with gr.Row(equal_height=True, elem_id="custom_row"):
        gr.HTML("""
        <div style="height: 400px; overflow: auto;">
        """)
        #input_row.scale = 1
        file_input = gr.File(label="Выберите файл", scale=10)
        input_image = gr.Image(label="Открытое изображение", elem_id="my_image", interactive=False, scale=10)
        redo_button = gr.Button("Сбросить ввод", elem_id="reset_button", scale=0)
          
    # Вставляем CSS через компонент HTML
    style = """
    <style>
    /* Для больших экранов */
    #my_image .gri-image {
        height: 600px  !important;
    }
    /* Для средних экранов */
    @media (max-width: 768px) {
        #my_image .gri-image {
            height: 400px  !important;
        }
    }
    /* Для маленьких экранов */
    @media (max-width: 480px) {
        #my_image .gri-image {
            height: 100px  !important;
        }
    }
    </style>
    """
    gr.HTML(style)
          

    # Изначально скрыт изображение
    input_image.visible = False
    
    file_input.change(handle_file, inputs=file_input, outputs=[file_input, input_image, data])
    redo_button.click(reset, inputs=None, outputs=[file_input, input_image, data])
    
    process_button = gr.Button("Сегментировать")

    gallery = gr.Gallery(label="Результаты", columns=3, interactive=False)
    text_feild = gr.Textbox(label=f"Result", elem_id="result_feild", interactive=False, lines=10)

    process_button.click(
        segmentate_data,
        inputs=[data],
        outputs=[text_feild, gallery]
    )
 

if __name__ == "__main__":
    demo.launch()
    
    