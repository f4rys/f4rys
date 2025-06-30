import os
import requests
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from mnists import MNIST

GITHUB_USERNAME = "f4rys"
GITHUB_TOKEN = os.environ.get("GH_TOKEN")
OUTPUT_IMAGE_NAME = "mnist_commits.png"

def get_total_commits():
    """Fetches the total number of commits for the user."""
    url = "https://api.github.com/graphql"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
    query = {
        "query": f"""
        query {{
            user(login: "{GITHUB_USERNAME}") {{
                contributionsCollection {{
                    contributionCalendar {{
                        totalContributions
                    }}
                }}
            }}
        }}
        """
    }

    response = requests.post(url, json=query, headers=headers)
    if response.status_code == 200:
        data = response.json()
        total_commits = data["data"]["user"]["contributionsCollection"]["contributionCalendar"]["totalContributions"]
        print(f"Total commits found: {total_commits}")
        return total_commits
    else:
        raise Exception(f"GitHub API query failed with status code: {response.status_code}\n{response.text}")

def generate_commit_image(commit_count):
    """
    Generates an image displaying the commit count using MNIST digits.
    """
    print("Loading MNIST dataset...")
    mnist = MNIST()
    x_train = mnist.train_images()
    y_train = mnist.train_labels()

    commit_str = str(commit_count)
    digit_images = []

    target_digit_height = 80
    target_digit_width = 80
    margin_x = 32
    margin_y = 20
    title_text = "Total Commits:"
    title_font_size = 50
    title_margin = 20
    min_width = 500
    min_height = 100

    # Try to load a truetype font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", title_font_size)
    except Exception:
        font = ImageFont.load_default()

    print(f"Generating images for commit count: {commit_str}")
    for digit in commit_str:
        digit = int(digit)
        indices = np.where(y_train == digit)[0]
        random_index = np.random.choice(indices)
        digit_image_array = x_train[random_index]
        digit_image = Image.fromarray(digit_image_array.astype('uint8'), 'L')
        # Upscale digit
        digit_image = digit_image.resize((target_digit_width, target_digit_height), resample=Image.NEAREST)
        # Invert digit to make it white on black
        #digit_image = Image.eval(digit_image, lambda px: 255 - px)
        digit_images.append(digit_image)

    digits_width = sum(img.width for img in digit_images)
    digits_height = target_digit_height
    total_width = max(digits_width + 2 * margin_x, min_width)

    # Calculate title size using textbbox
    dummy_img = Image.new('L', (10, 10))
    draw = ImageDraw.Draw(dummy_img)
    try:
        bbox = draw.textbbox((0, 0), title_text, font=font)
        title_w = bbox[2] - bbox[0]
        title_h = bbox[3] - bbox[1]
    except AttributeError:
        title_w, title_h = font.getsize(title_text)

    total_height = max(title_h + title_margin + digits_height + 2 * margin_y, min_height)

    # Create new image with black background
    combined_image = Image.new('L', (total_width, total_height), color=0)
    draw = ImageDraw.Draw(combined_image)

    # Draw title centered in white
    title_x = (total_width - title_w) // 2
    title_y = margin_y
    draw.text((title_x, title_y), title_text, font=font, fill=255)

    # Paste digits below title
    x_offset = (total_width - digits_width) // 2  # center digits horizontally
    y_offset = margin_y + title_h + title_margin

    for img in digit_images:
        combined_image.paste(img, (x_offset, y_offset))
        x_offset += img.width

    combined_image.save(OUTPUT_IMAGE_NAME)
    print(f"Successfully generated and saved image as {OUTPUT_IMAGE_NAME}")

if __name__ == "__main__":
    if not GITHUB_TOKEN:
        raise ValueError("GitHub token not found. Please set the GH_TOKEN environment variable.")
    total_commits = get_total_commits()
    generate_commit_image(total_commits)