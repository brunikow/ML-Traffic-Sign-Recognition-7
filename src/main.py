from Data.Data import ImageDataset
from torch.utils.data import DataLoader

def main():
    folderpath = "../data/GTSRB/Final_Training/Images/"
    filepath = "../data/concepts_per_class.csv"

    dataset = ImageDataset(folderpath, filepath, 128, 128)
    custom_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"Custom dataset size: {len(dataset)}")
    print(f"Number of batches: {len(custom_loader)}")

    # Show sample
    sample_img, label_id, label_string, concept_vector = dataset[0]
    print(f"Sample shape: {sample_img.shape}, Label: {label_id}, {label_string}, {concept_vector}")


if __name__ == "__main__":
    main()