import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Patch
from matplotlib.colors import Normalize

def read_flu_data(file_path):
    with open(file_path, 'r') as file:
        data = []
        current_day = []
        
        for line in file:
            if line.startswith("Day"):
                if current_day:
                    data.append(np.array(current_day))
                    current_day = []
            else:
                current_day.append(list(map(int, line.split())))

        # Append the last day if not added
        if current_day:
            data.append(np.array(current_day))
    
    return data

def create_legend(ax, data, cmap):
    unique_values = np.unique(data)
    norm = Normalize(vmin=unique_values.min(), vmax=unique_values.max())  # Normalize the color range
    colors = cmap(norm(unique_values))
    
    legend_patches = [
        Patch(color=colors[i], label=f"{int(value)}")
        for i, value in enumerate(unique_values)
    ]
    
    ax.legend(handles=legend_patches, loc='upper right', title="Legend", fontsize=8)

def animate_data(days):
    fig, ax = plt.subplots()
    cmap = plt.cm.coolwarm
    
    # Get global min and max from all frames to ensure consistent colormap scaling
    all_data = np.concatenate([day.flatten() for day in days])
    global_norm = Normalize(vmin=all_data.min(), vmax=all_data.max())
    
    # Initialize the first frame with global normalization
    im = ax.imshow(days[0], cmap=cmap, norm=global_norm)
    
    # Add text annotation for the day counter
    day_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white', fontsize=12)

    def update(frame):
        # Normalize each frame independently
        local_norm = Normalize(vmin=days[frame].min(), vmax=days[frame].max())
        im.set_data(days[frame])
        im.set_norm(local_norm)
        day_text.set_text(f'Day {frame}')
        return im, day_text

    ani = animation.FuncAnimation(
        fig, update, frames=len(days), blit=True, interval=1000
    )
    plt.show()

def plot_data(data):
    days = len(data)
    cols = int(np.ceil(np.sqrt(days)))
    rows = int(np.ceil(days / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

    # Flatten the axs array if there are multiple rows
    axs = axs.flatten() if rows > 1 else [axs]

    cmap = plt.cm.coolwarm  # Define colormap for consistent coloring

    for day_idx, (ax, day_data) in enumerate(zip(axs, data)):
        norm = Normalize(vmin=day_data.min(), vmax=day_data.max())  # Normalize based on each day's data range
        im = ax.imshow(day_data, cmap=cmap, interpolation='nearest', norm=norm)
        ax.set_title(f"Day {day_idx}")
        ax.set_aspect('equal')  # Make each plot square
        create_legend(ax, day_data, cmap)

    # Hide unused subplots in case of an uneven grid
    for ax in axs[len(data):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    flu_data = read_flu_data("flu_simulation.txt")
    plot_data(flu_data)
    animate_data(flu_data)

    