import torch
from tqdm import tqdm
from transformers.utils import logging
import os
import pickle

logger = logging.get_logger(__name__)


class Hook:
    def __init__(self, module, backward=False):
        self.calib = None
        if not backward:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        with torch.no_grad():
            inp = input[0].detach().float()
            inp = inp.flatten(start_dim=0, end_dim=-2)
            if self.calib is None:
                self.calib = (inp.T @ inp).cpu()
            else:
                self.calib += (inp.T @ inp).cpu()

    def close(self):
        self.hook.remove()
        del self.calib


class Calib:
    @staticmethod
    def save(path, data):
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def get_calib_data(group, name, save_path=None):
        assert save_path is not None
        # calibration data for up and gate is the same
        if name == 'mlp.gate_proj':
            name = 'mlp.up_proj'
        if name == "self_attn.q_proj" or name == "self_attn.v_proj":
            name = "self_attn.k_proj"
        data = None
        file_path = os.path.join(save_path, name)
        for item in group:
            file_name = os.path.join(file_path, "{}.pkl".format(item))
            if os.path.exists(file_name):
                raw = Calib.load(file_name)
                cov, _ = Calib._split_cov_scaling(raw)
                if cov is None:
                    raise ValueError(f"{file_name} missing covariance data")
                if data is None:
                    data = cov
                else:
                    data += cov
            else:
                raise FileNotFoundError(
                    "{} not found. You should run build_calibration_dataset first!".format(file_name))
        return data

    @staticmethod
    def get_s_inv_s(group, name, model_type, calib_path=None):
        data = Calib.get_calib_data(group, name, calib_path).double()
        scaling_diag_matrix, scaling_diag = Calib._compute_scaling_matrix(data)
        if calib_path is not None:
            Calib._save_scaling_diag(calib_path, name, group, scaling_diag)
        invs = torch.linalg.inv(scaling_diag_matrix)
        return scaling_diag_matrix, invs, scaling_diag

    @staticmethod
    def build_calibration_dataset(model, dataloader, names, model_type, save_path):
        print("Start building calibration data.")

        if model_type == "gpt2":
            tmp_model = model.transformer.h
        elif model_type == "llama2":
            tmp_model = model.model.layers
        elif model_type == "opt":
            tmp_model = model.model.decoder.layers
        elif model_type == "mistral":
            tmp_model = model.model.layers
        else:
            raise NotImplementedError

        hooks = {}
        for name in names:
            hooks[name] = []
            for layer in tmp_model:
                target = layer.get_submodule(name)
                hooks[name].append(Hook(target, backward=False))

        model.config.use_cache = False
        model.eval()
        for i, batch in tqdm(enumerate(dataloader)):
            with torch.no_grad():
                batch = {k: v.to(model.device) for k, v in batch.items()}
                out = model(**batch)

        assert save_path is not None
        for name in names:
            tmp_save_path = os.path.join(save_path, name)
            if not os.path.exists(tmp_save_path):
                os.makedirs(tmp_save_path)
            for i, hook in enumerate(hooks[name]):
                cov = hook.calib.cpu()
                scaling_diag = Calib._compute_scaling_diag(cov)
                data = {"cov": cov, "scaling_diag": scaling_diag}
                tmp_name = str(i) + ".pkl"
                Calib.save(os.path.join(tmp_save_path, tmp_name), data)
                hook.close()

    @staticmethod
    def build_update_dataset(model, dataloader, names, model_type, save_path):
        print("Start building update dataset.")
        if model_type == "gpt2":
            tmp_model = model.transformer
            num_layers = len(tmp_model.h)
        elif model_type == "llama2" or model_type == "mistral":
            tmp_model = model.model
            num_layers = len(tmp_model.layers)
        elif model_type == "opt":
            tmp_model = model.model.decoder
            num_layers = len(tmp_model.layers)
        else:
            raise NotImplementedError

        hooks = {}
        for name in names:
            hooks[name] = []
            for i in range(num_layers):
                target = tmp_model.get_submodule(name)[str(i)]
                hooks[name].append(Hook(target, backward=False))

        model.config.use_cache = False
        model.eval()
        for i, batch in tqdm(enumerate(dataloader)):
            with torch.no_grad():
                batch = {k: v.to(model.device) for k, v in batch.items()}
                out = model(**batch)

        assert save_path is not None
        for name in names:
            tmp_save_path = os.path.join(save_path, name)
            if not os.path.exists(tmp_save_path):
                os.makedirs(tmp_save_path)
            for i, hook in enumerate(hooks[name]):
                cov = hook.calib.cpu()
                scaling_diag = Calib._compute_scaling_diag(cov)
                data = {"cov": cov, "scaling_diag": scaling_diag}
                tmp_name = str(i) + ".pkl"
                Calib.save(os.path.join(tmp_save_path, tmp_name), data)
                hook.close()

    @staticmethod
    def _compute_scaling_matrix(data):
        # The following code is from https://github.com/AIoT-MLSys-Lab/SVD-LLM
        try:
            scaling_diag_matrix = torch.linalg.cholesky(data).T
        except Exception as e:
            print("Warning: eigen scaling_diag_matrix is not positive!")
            eigenvalues = torch.linalg.eigvalsh(data)
            data = data + (- eigenvalues[0] + 7e-6) * torch.eye(data.shape[0]).to(data.device)
            scaling_diag_matrix = torch.linalg.cholesky(data).T
            eigenvalues = None
            del eigenvalues
        scaling_diag = torch.diag(scaling_diag_matrix).float()
        return scaling_diag_matrix, scaling_diag

    @staticmethod
    def _compute_scaling_diag(data):
        _, scaling_diag = Calib._compute_scaling_matrix(data.double())
        return scaling_diag

    @staticmethod
    def _split_cov_scaling(raw):
        if isinstance(raw, dict):
            cov = raw.get("cov", None)
            scaling_diag = raw.get("scaling_diag", None)
        else:
            cov = raw
            scaling_diag = None
        return cov, scaling_diag

    @staticmethod
    def _save_scaling_diag(base_path, name, group, scaling_diag):
        dir_path = os.path.join(base_path, name, "scaling")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        group_name = "_".join([str(i) for i in group])
        file_path = os.path.join(dir_path, f"{group_name}.pkl")
        Calib.save(file_path, scaling_diag.cpu())
