import argparse
from dataclasses import dataclass, field

@dataclass
class Config:
    week: int                   = 1
    
    gpu_num: int                = 0
    seed: int                   = 64   # 64, 2025, 1024, 42, 256
    hop: int                    = 2
    interval_minutes: int       = 30       
    batch_size: int             = 300

    method: str                 = 'multihead_self_attention'   
    no_category: bool           = False
    unique_category: bool       = False
    adjust_score: bool          = True
    use_grnn: bool              = True

    num_words: int              = 1 + 330_899
    word_embedding_dim: int     = 100
    num_categories: int         = 26
    num_categories_for_NewsEncoder: int = 14
    num_subcategories_for_NewsEncoder: int = 55
    num_filters: int            = 100
    query_vector_dim: int       = 200
    window_size: int            = 3
    dropout_probability: float  = 0.2

    head_num: int               = 20
    head_dim: int               = 15
    dataset_lang: str           = 'norwegian'
    category_emb_dim: int       = 100
    subcategory_emb_dim: int    = 100
    attention_hidden_dim: int   = 100

    _parser: argparse.ArgumentParser = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        self._parser = argparse.ArgumentParser(
            description="Setting hyperparameters"
        )
        add = self._parser.add_argument

        add("--week",              type=int,   default=self.week,
            help="select which dataset of week")
        
        add("--gpu_num",           type=int,   default=self.gpu_num)
        add("--seed",              type=int,   choices=[64, 2025, 1024, 42, 256], default=self.seed)
        add("--hop",               type=int,   choices=[1, 2, 3], default=self.hop)
        add("--interval_minutes",  type=int,   choices=[30, 720, 1440, 2160], default=self.interval_minutes)
        add("--batch_size",        type=int,   default=self.batch_size)

        add("--method",            type=str,   default=self.method,
            choices=["cnn_attention", "multihead_self_attention"])
        add("--no_category",       action="store_true")   
        add("--unique_category",   action="store_true")
        add("--adjust_score",      action="store_true")
        add("--use_grnn",          action="store_true")

    def parse(self):
        args = self._parser.parse_args()

        for key, value in vars(args).items():
            setattr(self, key, value)

        return self
    
    def _apply_week(self):


        if self.week == 1:
            self.num_categories = 26
            self.num_categories_for_NewsEncoder = 14
            self.num_subcategories_for_NewsEncoder = 55
        elif self.week == 2:
            self.num_categories = 34
            self.num_categories_for_NewsEncoder = 16
            self.num_subcategories_for_NewsEncoder = 80
        else:
            self.num_categories = 35
            self.num_categories_for_NewsEncoder = 17
            self.num_subcategories_for_NewsEncoder = 93


        self.interval_minutes = 30 * self.week


def get_config() -> Config:
    """
    usage:
        from config import get_config
        cfg = get_config()
    """
    return Config().parse()
