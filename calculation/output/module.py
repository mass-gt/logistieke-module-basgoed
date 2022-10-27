import logging

import pandas as pd
from calculation.common.base_module import BaseModule
from calculation.common.io import read_mtx
import calculation.output.support as support
from settings import Settings


class OutputModule(BaseModule):
    """
    The output module, which generates output statistics based on outputs of the other modules
    of the WM.
    """

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        """
        Constructs the output module.
        Forwards the settings and logger references to the base module.
        Initializes variables to prepare the calculation of outputs.

        Args:
            settings (Settings): A reference to the settings.
            logger (logging.Logger): A reference to the logger.
        """
        BaseModule.__init__(self, settings, logger)

        self.logger.debug("\tInladen bestanden...")

        self.tours = pd.read_csv(
            settings.get_path("Tours"), sep='\t',
            usecols=['orig_nrm', 'dest_nrm', 'vehicle_type_lwm'])
        self.parcelTours = pd.read_csv(
            settings.get_path("PakketRondritten"), sep='\t',
            usecols=['orig_nrm', 'dest_nrm', 'vehicle_type_lwm'])

        self.tripsVanConstruction = read_mtx(settings.get_path("RittenBestelBouw"))
        self.tripsVanService = read_mtx(settings.get_path("RittenBestelService"))

        self.zones = pd.read_csv(settings.get_path("ZonesNRM"), sep='\t')

        self.lengthClassShares = support.get_length_class_shares(settings)

    def run(self):
        """
        Performs the actual calculations of the output module.
        """
        self.logger.debug("\tKoppeling NRM naar VAM leggen...")

        arrayNRMtoVAM = support.get_nrm_to_vam(self.zones)
        tours = support.add_vam_to_tours(self.tours, arrayNRMtoVAM)

        self.logger.debug("\tVrachtautomatrices opstellen...")

        matricesVAM = support.make_vam_matrices(
            tours, self.lengthClassShares, self.dimLengthClass,
            self.settings.module.year_factor_goods)

        self.logger.debug("\tVrachtautomatrices wegschrijven...")

        support.write_vam_matrices(self.settings, matricesVAM)

        self.logger.debug("\tBestelautomatrix opstellen...")

        matricesBAM = support.make_bam_matrices(
            self.tripsVanConstruction, self.tripsVanService, self.tours, self.parcelTours,
            arrayNRMtoVAM, self.dimVanSegment, self.dimVT,
            self.settings.module.year_factor_construction,
            self.settings.module.year_factor_service,
            self.settings.module.year_factor_goods,
            self.settings.module.year_factor_parcel)

        self.logger.debug("\tBestelautomatrix wegschrijven...")

        support.write_bam_matrix(self.settings, matricesBAM)
