from glob import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import seaborn as sns


def show_training_results():

    parents = glob('./runs/*')
    paths = list(map(lambda x: glob(f"{x}/events*")[0], parents))
    print(paths)
    for path in paths:
        event_acc = EventAccumulator(path, size_guidance={'scalars': 0})
        event_acc.Reload()

        scalars = {}
        for tag in event_acc.Tags()['scalars']:
            events = event_acc.Scalars(tag)
            scalars[tag] = [event.value for event in events]

        sns.set()

        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        plt.plot(range(len(scalars['Loss/Train'])), scalars['Loss/Train'], label="Train Loss")
        plt.plot(range(len(scalars['Loss/Val'])), scalars['Loss/Val'], label="Val Loss")
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train/Val Loss')

        plt.subplot(1, 2, 2)
        plt.plot(range(len(scalars['Train/JaccardIndex'])),
                scalars['Train/JaccardIndex'], label='Train/JaccardIndex')
        plt.plot(range(len(scalars['Val/JaccardIndex'])),
                scalars['Val/JaccardIndex'], label='Val/JaccardIndex')
        plt.legend()
        plt.ylabel('JaccardIndex')
        plt.xlabel('Epoch')
        plt.title('Train/Val JaccardIndex')
        plt.show()
