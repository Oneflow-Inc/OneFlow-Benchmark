from eval import Trainer
from config import get_args
if __name__ == "__main__":
    args=get_args()
    trainer = Trainer(args)
    trainer()