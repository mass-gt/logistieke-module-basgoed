import logging
from typing import Dict, List

from settings import DimensionSettings, ModuleSettings, Settings, ZoneSettings


class BaseModule():
    settings: Settings = None
    logger: logging.Logger = None

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger

        self.read_dimensions()

    def read_dimensions(self):
        # Dimensies
        self.dimEmployment = self.dimensionSettings.employment_category
        self.dimLengthClass = self.dimensionSettings.length_class
        self.dimVanSegment = self.dimensionSettings.van_segment

    @property
    def dimGG(self) -> List[int]:
        return self.settings.commodities

    @property
    def dimLS(self) -> List[str]:
        return self.dimensionSettings.logistic_segment

    @property
    def dimVT(self) -> List[Dict[str, str]]:
        return self.dimensionSettings.vehicle_type_lwm

    @property
    def dimSS(self) -> List[str]:
        return self.dimensionSettings.shipment_size

    @property
    def moduleSettings(self) -> ModuleSettings:
        return self.settings.module

    @property
    def dimensionSettings(self) -> DimensionSettings:
        return self.settings.dimensions

    @property
    def zoneSettings(self) -> ZoneSettings:
        return self.settings.zones

    def run(self):
        raise NotImplementedError()
