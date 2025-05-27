import sys, os, json
from datetime import datetime

sys.path.append(os.environ.get("PROJECT_ROOT"))
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import tensorflow as tf

keras = tf.keras

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from sklearn.metrics import accuracy_score
from ldf.models.model_ldf import LdfModel
import ldf.models.nn_define_models as dm

EarlyStopping = keras.callbacks.EarlyStopping

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))


class LdfScalarModel(LdfModel):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def setup_model(self, X=None, load_model=False):

        if isinstance(self.params, dict):
            # If it's already a dictionary, skip json.loads
            self.params = self.params
        elif isinstance(self.params, str):
            # If it's a string, attempt to load it
            self.params = json.loads(self.params)
        else:
            # Raise an error if neither str nor dict
            raise TypeError(f"Expected `self.params` to be a `str` or `dict`, but got {type(self.params)}")

        self.define_distn()

        self.load_model_ = load_model

        # setup
        if self.gpu == 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        # format output
        out_basename = self.model_name \
                       + f'_{self.error_metric}' \
                       + f'_{datetime.now().strftime("%Y%m%d%H%M%S")}_{self.hash}' \
                       + f'_do{round(self.do_val * 100)}' \
                       + f'_rm{self.reg_mode}' \
                       + f'_rv{round(self.reg_val * 100)}'

        # make output directory
        if self.output_dir is not None:
            self.output_dir = os.path.join(self.output_dir, out_basename)
            print(f'output dir: {self.output_dir}')
            os.makedirs(self.output_dir, exist_ok=True)

        args = {key: val for key, val in self.__dict__.items() if
                isinstance(val, int) or
                isinstance(val, str) or
                isinstance(val, float) or
                isinstance(val, bool)}
        # save params to external csv and json files
        if self.output_dir is not None:
            with open(os.path.join(self.output_dir, 'args.json'), "w") as outfile:
                json.dump(args, outfile, indent=2)

        # get prob_callback
        self.callbacks_prob_ = \
            dm.get_callbacks_prob(red_lr_on_plateau=self.red_lr_on_plateau,
                                  red_factor=self.red_factor,
                                  red_patience=self.red_patience,
                                  red_min_lr=self.red_min_lr,
                                  verbose=self.verbose,
                                  es_patience=self.es_patience)

        # load model
        if load_model:

            self.model_ = dm.load_model(self.load_json,
                                        self.load_weights,
                                        self.distn)

            self.history_ = None
            self.train_mean_nll_ = None
            self.train_nlls_ = None

        else:

            model, _ = \
                dm.define_model(input_dim=X.shape[1],
                                output_dim=self.distn.get_output_dim(),
                                distn=self.distn,
                                do_val=self.do_val,
                                learning_rate_training=self.learning_rate_training,
                                model_name=self.model_name,
                                reg_mode=self.reg_mode,
                                reg_val=self.reg_val)

            self.model_ = model

    def fine_tune_model(self,
                        X=None,
                        y=None,
                        validation_data=None,
                        use_jpgs=None,
                        batch_size=None,
                        train_epochs=None,
                        use_multiprocessing=None,
                        break_out_val=True):

        if (len(X) == 2) and (type(X) is list):
            X = X[0]

        # If validation data is provided
        if validation_data:
            self.history_ = self.model_.fit(
                X, y,
                epochs=train_epochs if train_epochs else self.train_epochs,
                batch_size=batch_size if batch_size else self.batch_size,
                verbose=self.verbose,
                workers=1,
                validation_data=validation_data,  # Pass validation data
                callbacks=self.callbacks_prob_,
                use_multiprocessing=False,
                shuffle=True
            )
        # Fallback to validation_split if validation_data is not provided
        elif break_out_val:
            self.history_ = self.model_.fit(
                X, y,
                epochs=train_epochs if train_epochs else self.train_epochs,
                batch_size=batch_size if batch_size else self.batch_size,
                verbose=self.verbose,
                workers=1,
                validation_split=0.1,
                callbacks=self.callbacks_prob_,
                use_multiprocessing=False,
                shuffle=True
            )
        else:
            self.history_ = self.model_.fit(
                X, y,
                epochs=train_epochs if train_epochs else self.train_epochs,
                batch_size=batch_size if batch_size else self.batch_size,
                verbose=self.verbose,
                callbacks=self.callbacks_prob_,
                use_multiprocessing=False,
                shuffle=True
            )

        return self.history_.history

    def fit(self, X=None, y=None, load_model=False):

        self.setup_model(X=X, y=y, load_model=load_model)

        self.fine_tune_model(X=X, y=y)

        return self

    def save_model(self, label):

        # save model definition as .json and model weights as .h5
        model_json = self.model_.to_json()  # serialize model to JSON
        json_path = os.path.join(self.output_dir, f'model_{label}.json')
        with open(json_path, 'w') as json_file:
            json_file.write(model_json)

        weights_path = os.path.join(self.output_dir, f'model_{label}.h5')
        self.model_.save_weights(weights_path)  # serialize weights to HDF5

        print('Saved model to disk')

        return json_path, weights_path

    def predict(self, X=None, convert_to_float=True):

        if convert_to_float:
            yhat = self.model_.predict(X.astype(np.float64), verbose=0)
        else:
            yhat = self.model_.predict(X, verbose=0)

        mean_preds, \
            std_preds, \
            preds_params, \
            preds_params_flat = \
            self.distn.interpret_predict_output(yhat)

        return mean_preds, std_preds, preds_params, preds_params_flat

    def predict_for_train(self,
                          train_x,
                          train_y,
                          get_acc=True):

        self.train_mean_preds, \
            _, \
            self.train_preds_params, \
            self.train_preds_params_flat = \
            self.predict(X=train_x)

        self.train_mean_nll, \
            self.train_nlls = \
            self.distn.compute_nll(self.train_preds_params,
                                   train_y)
        if get_acc:
            self.train_acc = accuracy_score(train_y, (self.train_mean_preds >= 0.5))

        return self

    def predict_for_val(self,
                        val_x,
                        val_y,
                        get_acc=True):

        self.val_mean_preds, \
            _, \
            self.val_preds_params, \
            self.val_preds_params_flat = \
            self.predict(X=val_x)

        self.val_mean_nll, \
            self.val_nlls = \
            self.distn.compute_nll(self.val_preds_params,
                                   val_y)
        if get_acc:
            self.val_acc = accuracy_score(val_y, (self.val_mean_preds >= 0.5))

        return self

    def predict_for_test(self,
                         test_x,
                         test_y,
                         get_acc=True):

        self.test_mean_preds, \
            _, \
            self.test_preds_params, \
            self.test_preds_params_flat = \
            self.predict(X=test_x)

        self.test_mean_nll, \
            self.test_nlls = \
            self.distn.compute_nll(self.test_preds_params,
                                   test_y)

        if get_acc:
            self.test_acc = accuracy_score(test_y, (self.test_mean_preds >= 0.5))

        return self

    def predict_for_train_chunks(self, data_generator, train_chunk_dir, batch_size, get_acc=True, max_chunks=2):
        """
        Predict and compute metrics for training data in chunks.
        """
        mean_preds = []
        nlls = []

        for x_train, y_train, _, _ in data_generator(train_chunk_dir, batch_size, max_chunks=max_chunks):
            preds, _, preds_params, _ = self.predict(X=x_train)
            mean_preds.append(preds)
            nll, nll_batch = self.distn.compute_nll(preds_params, y_train.flatten())
            nlls.append(nll_batch)

        # Combine results across chunks
        self.train_mean_preds = np.concatenate(mean_preds, axis=0)
        self.train_nlls = np.concatenate(nlls, axis=0)
        self.train_mean_nll = np.mean(self.train_nlls)

        if get_acc:
            self.train_acc = accuracy_score(y_train.flatten(), (self.train_mean_preds >= 0.5))

        return self

    def predict_for_val_chunks(self, data_generator, val_chunk_dir, batch_size, get_acc=True, max_chunks=2):
        """
        Predict and compute metrics for validation data in chunks.
        """
        mean_preds = []
        nlls = []

        for x_val, y_val, _, _ in data_generator(val_chunk_dir, batch_size, max_chunks=max_chunks):
            preds, _, preds_params, _ = self.predict(X=x_val)
            mean_preds.append(preds)
            nll, nll_batch = self.distn.compute_nll(preds_params, y_val.flatten())
            nlls.append(nll_batch)

        # Combine results across chunks
        self.val_mean_preds = np.concatenate(mean_preds, axis=0)
        self.val_nlls = np.concatenate(nlls, axis=0)
        self.val_mean_nll = np.mean(self.val_nlls)

        if get_acc:
            self.val_acc = accuracy_score(y_val.flatten(), (self.val_mean_preds >= 0.5))

        return self

    def predict_for_test_chunks(self, data_generator, test_chunk_dir, batch_size, get_acc=True, max_chunks=2):
        """
        Predict and compute metrics for test data in chunks.
        """
        mean_preds = []
        nlls = []
        y_test_all = []
        for x_test, y_test, _, _ in data_generator(test_chunk_dir, batch_size, max_chunks=max_chunks):
            preds, _, preds_params, _ = self.predict(X=x_test)
            mean_preds.append(preds)
            nll, nll_batch = self.distn.compute_nll(preds_params, y_test.flatten())
            nlls.append(nll_batch)
            y_test_all.append(y_test.flatten())

        # Combine results across chunks
        self.test_mean_preds = np.concatenate(mean_preds, axis=0)
        self.test_nlls = np.concatenate(nlls, axis=0)
        self.test_mean_nll = np.mean(self.test_nlls)

        if get_acc:
            self.test_acc = accuracy_score(y_test.flatten(), (self.test_mean_preds >= 0.5))

        return self, np.concatenate(y_test_all, axis=0)

    def save_summary(self,
                     data):

        self.metrics_df = pd.DataFrame(columns=['fold_num', 'metric', 'val'])

        # metrics to log for each fold
        metrics_to_log = [
            ('data', data),
            ('train_mean_preds', self.train_mean_preds),
            ('val_mean_preds', self.val_mean_preds),
            ('test_mean_preds', self.test_mean_preds),
            ('train_mean_nll', self.train_mean_nll),
            ('val_mean_nll', self.val_mean_nll),
            ('test_mean_nll', self.test_mean_nll),
            ('train_median_nll', np.median(self.train_nlls)),
            ('val_median_nll', np.median(self.val_nlls)),
            ('test_median_nll', np.median(self.test_nlls))
        ]
        if 'train_preds_params' in self.__dict__:
            metrics_to_log.append(('train_preds_params', self.train_preds_params))
        if 'val_preds_params' in self.__dict__:
            metrics_to_log.append(('val_preds_params', self.val_preds_params))
        if 'test_preds_params' in self.__dict__:
            metrics_to_log.append(('test_preds_params', self.test_preds_params))

        if hasattr(self, 'history_'):
            if self.history_ is not None:
                if hasattr(self.history_, 'history'):
                    metrics_to_log.append(('history_loss', self.history_.history['loss']))
                else:
                    metrics_to_log.append(('history_loss', self.history_['history']['loss']))
                if hasattr(self.history_, 'history'):
                    metrics_to_log.append(('history_val_loss', self.history_.history['val_loss']))
                else:
                    metrics_to_log.append(('history_val_loss', self.history_['history']['val_loss']))
        if 'train_acc' in self.__dict__:
            metrics_to_log.append(('train_acc', self.train_acc))
        if 'val_acc' in self.__dict__:
            metrics_to_log.append(('val_acc', self.val_acc))
        if 'test_acc' in self.__dict__:
            metrics_to_log.append(('test_acc', self.test_acc))

        for run_metric_name, run_metric_val in metrics_to_log:
            metric_dict = {
                'subset': '',
                'metric': run_metric_name,
                'val': run_metric_val
            }
            self.metrics_df = pd.concat([self.metrics_df, pd.DataFrame([metric_dict])], ignore_index=True)

        self.metrics_df.to_pickle(os.path.join(self.output_dir, 'metrics_df.p'))
        self.metrics_df.to_csv(os.path.join(self.output_dir, 'metrics_df.csv'))

        return self

    def return_metrics_df(self):

        return self.metrics_df

    def get_output_dir(self):
        return self.output_dir
