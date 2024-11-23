from utils.dataset import COCO8Dataset, CarportsDataset, create_dataloader

def test_dataset():
    coco8_train_dataset = COCO8Dataset(
        root_path="datasets/coco8-seg",
        split='train',
        img_size=640
    )

    coco8_val_dataset = COCO8Dataset(
        root_path="datasets/coco8-seg",
        split='val',
        img_size=640
    )

    carports_train_dataset = CarportsDataset(
        root_path="datasets/carparts-seg",
        split='train',
        img_size=640
    )

    carports_val_dataset = CarportsDataset(
        root_path="datasets/carparts-seg",
        split='valid',
        img_size=640
    )

    coco8_train_loader = create_dataloader(coco8_train_dataset, batch_size=64, shuffle=True)
    coco8_val_loader = create_dataloader(coco8_val_dataset, batch_size=64, shuffle=False)
    carports_train_loader = create_dataloader(carports_train_dataset, batch_size=64, shuffle=True)
    carports_val_loader = create_dataloader(carports_val_dataset, batch_size=64, shuffle=False)

    print(f"COCO8 Train: {len(coco8_train_dataset)} images, {len(coco8_train_loader)} batches")
    print(f"COCO8 Val: {len(coco8_val_dataset)} images, {len(coco8_val_loader)} batches")
    print(f"Carports Train: {len(carports_train_dataset)} images, {len(carports_train_loader)} batches")
    print(f"Carports Val: {len(carports_val_dataset)} images, {len(carports_val_loader)} batches")

    return {
        coco8_train_loader,
        coco8_val_loader
    }

if __name__ == "__main__":
    test_dataset()


