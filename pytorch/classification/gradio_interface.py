import gradio as gr
import torch
import numpy as np

from src.metric import get_loader_by_namelist, get_accuracy_with_output
from src.models import Custom_VGG, Custom_ResNet, Custom_EfficientNet, Custom_DenseNet

def read_image(image_path, device="cpu"):
    # Место для чтения и обработки изображения
    # например, изменение яркости и контраста
    data = [data for data in get_loader_by_namelist([image_path], 1, device=device)]

    imgs, classes = data[0]
    targets = [int(classes[0].item())]
    
    img = imgs[0].permute(1, 2, 0).numpy()
    
    img = img[:,:,:3].astype(int)
    minv = img.min()
    maxv = img.max()
    img = (img - minv)/(maxv-minv) *255 if img.max() != img.min() else img

    return img.astype(np.uint8), data, targets


def init_models():
    model_vgg = Custom_VGG(ipt_size=(240, 240, 4), pretrained=False)
    model_vgg.load_state_dict(torch.load("model_vgg_11.pth"))

    model_resnet = Custom_ResNet(ipt_size=(240, 240, 4), pretrained=False)
    model_resnet.load_state_dict(torch.load("model_resnet.pth"))

    model_efficientnet = Custom_EfficientNet(ipt_size=(240, 240, 4), pretrained=False)
    model_efficientnet.load_state_dict(torch.load("model_efficientnet.pth"))

    model_densenet = Custom_DenseNet(ipt_size=(240, 240, 4), pretrained=False)
    model_densenet.load_state_dict(torch.load("model_densenet.pth"))

    return (("model_vgg",          model_vgg),
            ("model_resnet",       model_resnet),
            ("model_efficientnet", model_efficientnet),
            ("model_densenet",     model_densenet))


def get_accyracy_and_output(model_list, test_data, device="cpu"):
    name_list = []
    outputs_list = []
    accuracy_list = []

    for name_model, model in model_list:
        accuracy, outputs = get_accuracy_with_output(test_data, model, device=device)
        name_list.append(name_model)
        outputs_list.append(outputs)
        accuracy_list.append(accuracy)

    return name_list, accuracy_list, outputs_list

def calculate_hybrid_res(outputs_list, true_res, threshold_of_hybrid):
    def Hybrid_solution(outputs):
        hybrid_res = np.array(outputs)
        return hybrid_res.mean(axis=0)

    def threshold_val(hybrid_res, threshold):
        hybrid_res[hybrid_res<threshold] = 0
        hybrid_res[hybrid_res>0] = 1
        return hybrid_res.astype(int)

    hybrid_res = Hybrid_solution(outputs_list)
    n = len(true_res)
    tp = (np.array(threshold_val(hybrid_res, threshold_of_hybrid)) == np.array(true_res)).sum()

    return hybrid_res, tp / n


def handle_file(file_name):
    img, data, targets = read_image(file_name)
    # После выбора файла скрыть input и показать изображение
    return gr.update(visible=False), gr.update(visible=True, value=img), data, targets

def reset():
    # Обнуляем значение file_input и показываем его
    return gr.update(value=None, visible=True), gr.update(value=None, visible=False), None, None

# Обработка при нажатии кнопки
def classifiacate_data(model_list, test_data, true_target, threshold_of_hybrid):
    
    name_list, accuracy_list, outputs_list = get_accyracy_and_output(model_list, test_data, device="cpu")
    output_hybrid_res, accuracy_hybrid_res = calculate_hybrid_res(outputs_list, true_target, threshold_of_hybrid)

    str_result = f"True class: {true_target}\n\n"
    for i in range(len(name_list)):
        str_result += f'"{name_list[i]}": output = {outputs_list[i]} and accuracy "{accuracy_list[i]}"\n'
    str_result += "\n"
    str_result += f'Hibrid res: output = {output_hybrid_res} and accuracy "{accuracy_hybrid_res}"\n'
    str_result += f"\n{'No tumor' if output_hybrid_res[0] < threshold_of_hybrid else 'There is a tumor'}"
    return gr.update(value=str_result)


with gr.Blocks(fill_height=True) as demo:
    data = gr.State()
    targets = gr.State()
    models = gr.State(value=init_models())

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
    
    file_input.change(handle_file, inputs=file_input, outputs=[file_input, input_image, data, targets])
    redo_button.click(reset, inputs=None, outputs=[file_input, input_image, data, targets])
    
    with gr.Row():
        threshold_of_hybrid = gr.Number(value=0.5, label="Порог гибридной модели")
        #param2 = gr.Slider(0.1, 3.0, value=1.0, step=0.1, label="Пусть будет 2")
    
    process_button = gr.Button("Запуск классификации")

    text_feild = gr.Textbox(label=f"Result", elem_id="result_feild", interactive=False, lines=10)

    process_button.click(
        classifiacate_data,
        inputs=[models, data, targets, threshold_of_hybrid],
        outputs=[text_feild]
    )
 

if __name__ == "__main__":
    demo.launch()
    
    