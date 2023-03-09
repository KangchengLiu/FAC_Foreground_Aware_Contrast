'''

@author: Dr. Kangcheng Liu

'''

from models.base_ssl3d_model import BaseSSLMultiInputOutputModel

def build_model(model_config, logger):
    return BaseSSLMultiInputOutputModel(model_config, logger)


__all__ = ["BaseSSLMultiInputOutputModel", "build_model"]
