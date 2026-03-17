import torch

print(f"¿Cuda disponible?: {torch.cuda.is_available()}")
print(f"Dispositivo actual: {torch.cuda.get_device_name(0)}")
print(f"Versión de PyTorch: {torch.version.__version__}")

# Una pequeña prueba de fuerza
x = torch.rand(5, 3).cuda()
print("¡Prueba de cálculo en GPU exitosa!")