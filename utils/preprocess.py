from PIL import Image
from torchvision import transforms
import io

# Transformation identique au notebook de votre binôme
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def preprocess_image(image_input):
    """
    Accepte :
    - un fichier Flask (request.files)
    - des bytes base64 décodés
    - un chemin de fichier string
    Retourne : tensor (1, 1, 224, 224)
    """
    if isinstance(image_input, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_input)).convert('RGB')
    elif isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB')
    else:
        img = Image.open(image_input).convert('RGB')

    tensor = transform(img)
    return tensor.unsqueeze(0)  # (1, 1, 224, 224)