import utils as ut
import logging

if __name__ == '__main__':    
    ut.setup_logging()

    try:
        ut.log_message(f"Initiallizing model training", level=logging.INFO)
        config = ut.load_config(file_path='config.ini')
        pre = ut.Preprocess(config['Settings']['data_path'])
        pre.scale_df()
        x_train = pre.split_data(int(config['Settings']['train_split']), config['Settings']['train_label'])
        x_test = pre.split_data(int(config['Settings']['test_split']), config['Settings']['test_label'])
        ae = ut.autoencoder(x_train, x_test)
        ae.create_model()
        ae.compile_model()
        ae.fit_model()
        ae.plot_model()
        ae.save_model(config['Settings']['model_name'])

    except Exception as e:
        ut.log_message(f"The following occured: {str(e)}", level=logging.ERROR)