import json


# TODO better json handling
class BertJobConfiguration:
    def __init__(self, config_file_path):
        self.config = json.load(open(config_file_path, 'r', encoding='utf-8'))

    # TODO improve this implementation
    def replace_path_placeholders(self, files_location):
        self.config['data']['datasets'] = {key: value.replace('placeholder/', files_location)
                                      for (key, value) in self.config['data']['datasets'].items()}
        self.config['validation']['path'] = self.config['validation']['path'].replace('placeholder/', files_location)

    def get_name(self):
        return self.config['name']

    def get_token_file_type(self):
        return self.config["bert_token_file"]

    def get_model_file_type(self):
        return self.config["bert_model_file"]

    def get_learning_rate(self):
        return self.config["training"]["learning_rate"]

    def get_warmup_proportion(self):
        return self.config["training"]["warmup_proportion"]

    def get_total_training_steps(self):
        return self.config["training"]["total_training_steps"]

    def get_total_epoch_count(self):
        return self.config["training"]["num_epochs"]

    def get_num_workers(self):
        return self.config['training']['num_workers']

    def get_validation_folder_path(self):
        return self.config['validation']['path']

    def get_wiki_pretrain_dataset_path(self):
        return self.config["data"]["datasets"]['wiki_pretrain_dataset']

    def get_book_corpus_pretrain_dataset_path(self):
        return self.config["data"]["datasets"]['bc_pretrain_dataset']

    def get_decay_rate(self):
        return self.config["training"]["decay_rate"]

    def get_decay_step(self):
        return self.config["training"]["decay_step"]

    def get_model_config(self):
        return self.config["bert_model_config"]
