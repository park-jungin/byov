import os
import json

from pytorch_lightning import Trainer
from utils.config import argparser
from utils.util import CustomModelCheckpoint
from video_tasks import VideoAlignment


def main():
    task = VideoAlignment(args)

    custom_checkpoint_callback = CustomModelCheckpoint(
        every_n_epochs=args.save_every,
        filename="{epoch}",
        save_top_k=-1,
    )

    trainer = Trainer(
        # devices=args.num_gpus,
        devices=[0],
        accelerator="gpu",
        callbacks=custom_checkpoint_callback,
        max_epochs=args.epochs,
        default_root_dir=args.output_dir,
        log_every_n_steps=4
    )

    if args.eval_only:
        # trainer.validate(task, ckpt_path=args.ckpt)
        trainer.test(task, ckpt_path=args.ckpt)
    else:
        trainer.fit(task)


if __name__ == '__main__':
    args = argparser.parse_args()
    args.output_dir = os.path.join('./logs/exp_'+args.dataset, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f'{args.output_dir}/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    main()


# ssh -L 8878:localhost:6009 diml@165.132.57.160
# ssh -L 8888:localhost:6008 diml@165.132.57.160
# tensorboard --logdir './logs/exp_break_eggs/bestconfig/lightning_logs/version_57' --port=6008
# tensorboard --logdir './logs/exp_pour_liquid/bestconfig/lightning_logs/version_8' --port=6008

# tensorboard --logdir './logs/exp_pour_milk/with_em/lightning_logs/version_0' --port=6009
# tensorboard --logdir './logs/exp_pour_milk/bestconfig/lightning_logs/version_22' --port=6009
# tensorboard --logdir './logs/exp_tennis_forehand/bestconfig/lightning_logs/version_6' --port=6009
# tensorboard --logdir './logs/exp_tennis_forehand/bestconfig/lightning_logs/version_4' --port=6009