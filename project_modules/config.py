
class document_selection_model_config():
    model_path = r"./models/twin-albert-for-long-text-pair-classification/pytorch_model.bin"
    model_dir = r"./models/twin-albert-for-long-text-pair-classification/"
    batch_size = 12
    doc_overlap_length = 32


class sentence_selection_model_config():
    model_path = r"./models/albert-base-v2-fine-tuned-qnli-sample/checkpoint-2100"
    max_length = 512
    batch_size = 12


class answer_retrieval_model_config():
    model_path = r"./models/albert-base-v2-fine-tuned-squad-sample"
    max_length = 512
    batch_size = 12
    max_answer_length = 32
    answer_retrieval_threshold = 20


class project_config():
    upload_folder = './static/uploaded_documents'
    allowed_extensions = ['txt']
    log_dir = "./user_log/"
    model_log_dir = "./models_log/"


class data_config():
    qnli_training_data_path = r"E:\MyFiles\WorkSpace\Data\QNLIv2\QNLI"
    squadv2_training_data_path = r"E:\MyFiles\WorkSpace\Data\SQUAD2"


class flask_config():
    template_auto_reload = True
