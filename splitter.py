import splitfolders # Instala con: pip install split-folders

input_folder = r'C:\Users\oscar\Downloads\archive\leaves\Grape'
output_folder = 'data_final'

splitfolders.ratio(input_folder, output=output_folder, 
                   seed=1337, ratio=(.8, .2), # 80% entrenamiento, 20% validación
                   group_prefix=None)