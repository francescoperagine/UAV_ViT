import os
import rasterio
from rasterio.mask import mask
from pyshp import shapefile

# Percorso al file dell'ortomosaico (aggiorna con il tuo percorso)
orthophoto_path = './orthophoto.tif'

# Percorso al file dello shapefile (aggiorna con il tuo percorso)
shapefile_path = './all_plots.shp'

# Percorso alla cartella di output per i file PNG (aggiorna con il tuo percorso)
output_folder = './plots'

# Crea la cartella di output se non esiste
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Apri il file dello shapefile
sf = shapefile.Reader(shapefile_path)

# Apri il file dell'ortomosaico utilizzando rasterio
ortho_dataset = rasterio.open(orthophoto_path)

# Loop attraverso i record dello shapefile
for plot in sf.iterShapeRecords():
    # Ottieni la geometria del plot dal record
    plot_geometry = plot.shape.__geo_interface__
    interface = plot.__geo_interface__
    print(interface)

    # Ottieni il nome del plot dal record
    plot_name = plot.record[1]
    print(plot.record)

    # Leggi il subset dell'immagine dell'ortomosaico corrispondente al plot
    cropped_image, _ = mask(ortho_dataset, [plot_geometry], crop=True)
    print("image shape ", cropped_image.shape)

    # Controlla se l'immagine è nel formato RGBA (4 canali) e riduci a RGB (3 canali) se necessario
    if cropped_image.shape[0] == 4:
        cropped_image = cropped_image[:3, :, :]

    # Crea il percorso completo per il file PNG di output
    output_path = os.path.join(output_folder, f'{plot_name}.png')

    # Scrivi l'immagine ritagliata come file PNG utilizzando rasterio
    with rasterio.open(output_path, 'w', driver='PNG', height=cropped_image.shape[1], width=cropped_image.shape[2], count=3, dtype='uint8') as dst:
        dst.write(cropped_image)
