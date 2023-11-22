from classic_methods import dict_classic_methods
from sota import sota_dict
from .example import NetExamples

dict_model = dict(
    example=NetExamples,
    **sota_dict,
    **dict_classic_methods
)
