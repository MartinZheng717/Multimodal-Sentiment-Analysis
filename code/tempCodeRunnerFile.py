if len(args.wandb_id) != 0:
    wandb.init(
        project="Text Sentiment Analysis using " + args.text_model,
        config=args,
        entity=args.wandb_id
    )