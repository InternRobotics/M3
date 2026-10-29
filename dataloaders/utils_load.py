from dataloaders.DatasetSelfCaptured import SelfCapturedDataset
from dataloaders.DatasetColmap import ColmapDataset

def load_dataset(args):
    if args.dataset_name == "selfCaptured":
        dataset = SelfCapturedDataset(args)
    elif args.dataset_name == "colmap":
        dataset = ColmapDataset(args)
    else:
        raise Exception("Dataset not found.")
    return dataset