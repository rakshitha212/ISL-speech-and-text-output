from PIL import Image, ImageDraw, ImageFont
import os

def generate_letter_gif(letter, filename):
    """Generate a sign language placeholder image for a letter."""
    # Color palette for each letter
    colors = {
        'a': (220, 50, 50), 'b': (50, 120, 220), 'c': (50, 180, 100),
        'd': (200, 150, 50), 'e': (150, 50, 200), 'f': (50, 200, 200),
        'g': (200, 100, 150), 'h': (100, 200, 50), 'i': (200, 80, 80),
        'j': (80, 80, 200), 'k': (80, 200, 80), 'l': (200, 200, 80),
        'm': (200, 80, 200), 'n': (80, 200, 200), 'o': (200, 130, 50),
        'p': (130, 50, 200), 'q': (50, 200, 130), 'r': (200, 50, 130),
        's': (50, 130, 200), 't': (130, 200, 50), 'u': (200, 100, 50),
        'v': (50, 100, 200), 'w': (100, 200, 50), 'x': (200, 50, 100),
        'y': (50, 200, 100), 'z': (100, 50, 200)
    }
    color = colors.get(letter.lower(), (80, 120, 200))

    img = Image.new('RGB', (200, 200), color=color)
    d = ImageDraw.Draw(img)

    # Draw border
    d.rectangle([4, 4, 195, 195], outline='white', width=4)

    # Draw letter large in center
    try:
        fnt_large = ImageFont.truetype("arial.ttf", 100)
        fnt_small = ImageFont.truetype("arial.ttf", 18)
    except IOError:
        fnt_large = ImageFont.load_default()
        fnt_small = ImageFont.load_default()

    letter_upper = letter.upper()
    bbox = d.textbbox((0, 0), letter_upper, font=fnt_large)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((200 - tw) / 2, (200 - th) / 2 - 15), letter_upper, font=fnt_large, fill='white')
    d.text((10, 170), f"ISL: {letter_upper}", font=fnt_small, fill='white')

    os.makedirs('static/gestures', exist_ok=True)
    img.save(f'static/gestures/{filename}', 'GIF')
    print(f"Generated: static/gestures/{filename}")

def generate_word_gif(word, filename):
    """Generate a placeholder GIF for a full word."""
    img = Image.new('RGB', (400, 200), color=(50, 80, 120))
    d = ImageDraw.Draw(img)
    d.rectangle([4, 4, 395, 195], outline='white', width=3)
    try:
        fnt = ImageFont.truetype("arial.ttf", 30)
        fnt_small = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        fnt = ImageFont.load_default()
        fnt_small = ImageFont.load_default()

    d.text((20, 20), "ISL Sign:", font=fnt_small, fill=(200, 200, 200))
    bbox = d.textbbox((0, 0), word, font=fnt)
    tw = bbox[2] - bbox[0]
    d.text(((400 - tw) / 2, 90), word, font=fnt, fill='white')

    os.makedirs('static/gestures', exist_ok=True)
    img.save(f'static/gestures/{filename}', 'GIF')
    print(f"Generated: static/gestures/{filename}")

if __name__ == '__main__':
    # Generate all alphabet letters
    import string
    for letter in string.ascii_lowercase:
        generate_letter_gif(letter, f'{letter}.gif')

    # Generate common words
    words = {
        'hello': 'hello.gif',
        'thanks': 'thanks.gif',
        'yes': 'yes.gif',
        'no': 'no.gif',
        'good morning': 'good_morning.gif',
        'good night': 'good_night.gif',
        'please': 'please.gif',
        'sorry': 'sorry.gif',
        'help': 'help.gif',
        'water': 'water.gif',
        'food': 'food.gif',
        'i love you': 'i_love_you.gif',
    }
    for word, fname in words.items():
        generate_word_gif(word.title(), fname)

    # Not found fallback
    img = Image.new('RGB', (300, 200), color=(60, 60, 60))
    d = ImageDraw.Draw(img)
    try:
        fnt = ImageFont.truetype("arial.ttf", 18)
    except IOError:
        fnt = ImageFont.load_default()
    d.text((60, 90), "Sign Not Found", font=fnt, fill=(200, 200, 200))
    img.save('static/gestures/not_found.gif', 'GIF')
    print("Generated: static/gestures/not_found.gif")
    print("\nAll gesture images generated!")
