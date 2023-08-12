import os
import rasterio
from rasterio.mask import mask
from lib.pyshp import shapefile

class Clipper:

    def __init__(self, orthophoto_path, shapefile_path, output_path):
        self.ORTHOMOSAIC_PATH = orthophoto_path
        self.SHAPEFILE_PATH = shapefile_path
        self.OUTPUT_PATH = output_path
        self.make_folder()

    # Crea la cartella di output se non esiste
    def make_folder(self):
        if not os.path.exists(self.OUTPUT_PATH):
            os.makedirs(self.OUTPUT_PATH)

    def start(self):
        # Apri il file dello shapefile
        sf = shapefile.Reader(self.SHAPEFILE_PATH)

        # Apri il file dell'ortomosaico utilizzando rasterio
        ortho_dataset = rasterio.open(self.ORTHOMOSAIC_PATH)

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
            file_path = os.path.join(self.OUTPUT_PATH, f'{plot_name}.png')

            # Scrivi l'immagine ritagliata come file PNG utilizzando rasterio
            with rasterio.open(file_path, 'w', driver='PNG', height=cropped_image.shape[1], width=cropped_image.shape[2], count=3, dtype='uint8') as dst:
                dst.write(cropped_image)


if __name__ == "__main__":
    orthomosaic_path = './data/orthophoto/raster.tif'
    shapefile_path = './data/raw/Case_Study_1/Shapefile/Plots_Shapefile/all_plots.shp'
    output_path = './data/plots'
    
    clipper = Clipper(orthomosaic_path, shapefile_path, output_path)
    clipper.start()
