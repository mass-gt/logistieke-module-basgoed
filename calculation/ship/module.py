import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from calculation.common.base_module import BaseModule
from calculation.common.io import get_skims
from calculation.common.params import set_seed
import calculation.ship.support as support
from calculation.ship.types import (CostFigure, CostFigures, Routes, Terminals, Tonnes)
from settings import Settings


class ShipModule(BaseModule):
    """
    The shipment module, which...
    """

    dimAccessEgressGG: List[int] = None
    dimAccessGG: List[int] = None

    employmentDict: Dict[str, int] = None

    truckCapacities: np.ndarray = None

    makeDistribution: np.ndarray = None
    useDistribution: np.ndarray = None

    cumProbsCommodityToNSTR: Tuple[np.ndarray, np.ndarray] = None
    probsCommodityToLS: Tuple[np.ndarray, np.ndarray] = None

    dictBGtoNRM: Dict[int, np.ndarray] = None
    dictCKtoNRM: Dict[int, np.ndarray] = None
    dictCKtoBG: Dict[int, int] = None
    dictNRMtoBG: Dict[int, int] = None

    distributionCenters: pd.DataFrame = None
    nrmZones: pd.DataFrame = None
    terminals: Terminals = None
    dcFractions: pd.DataFrame = None
    tonnes: Tonnes = None
    routes: Routes = None
    skim: Any = None
    costFigures: CostFigures = None

    coeffsShip: Dict[str, np.ndarray] = None
    coeffsDistDecay: Any = None

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        """
        Constructs the shipment module.
        Forwards the settings and logger references to the base module.
        Initializes many variables to prepare the calculation of shipments.

        Args:
            settings (Settings): A reference to the settings.
            logger (logging.Logger): A reference to the logger.
        """
        BaseModule.__init__(self, settings, logger)

        self.logger.debug("\tInladen bestanden...")

        self.employmentDict = support.get_employment(settings)
        self.truckCapacities = support.get_truck_capacities(self.settings, len(self.dimVT))

        self.makeDistribution, self.useDistribution = support.get_make_use(
            self.settings, self.employmentDict)
        self.cumProbsCommodityToNSTR = support.get_gg_to_nstr(self.settings)
        self.probsCommodityToLS = support.get_gg_to_ls(self.settings)

        self.nrmZones = support.get_nrm_zones(self.settings)

        self.dictBGtoNRM, self.dictCKtoNRM = support.get_dict_to_nrm(settings, self.nrmZones)
        self.dictCKtoBG, self.dictNRMtoBG = support.get_dict_to_bg(settings, self.nrmZones)

        self.distributionCenters = support.get_distribution_centers(self.settings, self.nrmZones)

        self.terminals = Terminals._make(
            (support.get_terminals_ck(self.settings, self.dictCKtoBG),) +
            support.get_terminals_ms(self.settings))

        support.update_zones_by_terminals_and_distribution_centers(
            self.nrmZones, self.distributionCenters, self.terminals.ck)

        self.dcFractions = support.get_fractions(self.settings)
        self.tonnes = Tonnes._make(
            support.get_tonnes_ms(self.settings, self.probsCommodityToLS, self.dimGG, self.dimLS) +
            (support.get_tonnes_ck(
                self.settings, self.probsCommodityToLS, self.dimGG, self.dimLS),))
        self.routes = Routes._make(
            support.get_routes_ck(self.settings, self.dimGG, self.terminals.ck))
        self.skim = np.c_[get_skims(
            self.settings, self.logger, self.nrmZones,
            freight=True, changeZeroValues=True, returnSkimCostCharge=True)]

        self.costFigures = CostFigures(
            CostFigure._make(support.get_cost_freight_vt(self.settings)),
            CostFigure._make(support.get_cost_freight_ms(self.settings)),
            CostFigure._make(support.get_cost_freight_ck(self.settings)))

        self.coeffsShip = support.get_coeffs_ship(
            self.settings, max(self.dimGG), len(self.dimSS), len(self.dimVT) - 2)

        self.coeffsDistDecay = support.get_coeffs_dist_decay(self.settings)

    def read_dimensions(self):
        """
        Goederengroepen waarvoor voor- en of natransport van/naar terminals wordt verondersteld
        voor niet-containerstromen
        """
        super().read_dimensions()

        self.dimAccessEgressGG = [
            int(x) for x in self.settings.dimensions.commodities_access_egress]
        self.dimAccessGG = [
            int(x) for x in self.settings.dimensions.commodities_access] + self.dimAccessEgressGG

    def run(self):
        """
        Performs the actual runs of the shipment module.
        """
        containerShipments = self.create_container_shipments()
        directShipments, railShipments, waterShipments = self.create_non_container_shipments()

        shipments = support.combine_shipments(
            containerShipments, directShipments, railShipments, waterShipments)

        shipments = support.apply_zez_zones(
            self.settings, self.logger, shipments, self.nrmZones, self.dictNRMtoBG)

        support.write_shipments_to_file(self.settings, self.logger, shipments)

        support.write_shipments_to_shp(self.settings, self.logger, shipments, self.nrmZones)

    def create_container_shipments(self) -> pd.DataFrame:
        """
        Creates container shipments.

        Returns:
            DataFrame: All container shipments.
        """
        containerStatistics = support.calculate_container_stats(self.settings)

        probabilities = support.calculate_container_probabilities(
            self.nrmZones, self.makeDistribution, self.useDistribution, self.dictCKtoNRM,
            self.employmentDict.keys(), self.dimGG)

        seed = set_seed(self.settings.module.seed_ship_ck, self.logger, 'seed-ship-ck')

        self.logger.debug("\tAanmaken zendingen gecontaineriseerd vervoer - direct...")

        shipments = support.create_container_shipments(
            self.tonnes.ck, self.routes, probabilities,
            self.distributionCenters, self.terminals.ck, self.dcFractions,
            self.coeffsDistDecay, containerStatistics, self.costFigures.container, self.skim,
            self.dictCKtoNRM, self.dictCKtoBG, self.zoneSettings.maximum_dutch_ck,
            self.cumProbsCommodityToNSTR, seed, self.moduleSettings.ship_discr_fac_ck,
            self.logger)

        self.logger.debug(f'\t\t{len(shipments)} containerzendingen aangemaakt.')

        return shipments

    def create_non_container_shipments(self) -> Tuple[pd.DataFrame]:
        """
        Creates all non-container shipments.

        Returns:
            Tuple[DataFrame]: Three collections of shipments.
        """
        probabilities = support.calculate_non_container_probabilities(
            self.nrmZones, self.makeDistribution, self.useDistribution, self.dictBGtoNRM,
            self.employmentDict.keys(), self.dimGG)

        seed = set_seed(
            self.settings.module.seed_ship_ms_direct, self.logger, 'seed-ship-ms-direct')

        self.logger.debug("\tAanmaken zendingen niet-gecontaineriseerd vervoer - direct...")

        shipmentsDirect = support.create_non_container_shipments_direct(
            self.tonnes.road, probabilities, self.distributionCenters,
            self.dcFractions, self.coeffsShip, self.coeffsDistDecay, self.skim,
            self.costFigures.vehicleType, self.costFigures.nonContainer, self.truckCapacities,
            self.dictBGtoNRM, self.zoneSettings.maximum_dutch, self.cumProbsCommodityToNSTR,
            self.dimVT, self.dimSS, seed, self.moduleSettings.ship_discr_fac_ms_direct,
            self.logger)

        self.logger.debug(
            f'\t\t{len(shipmentsDirect)} directe zendingen niet-container aangemaakt.')

        self.logger.debug(
            "\tAanmaken zendingen niet-gecontaineriseerd vervoer - spoorterminals...")

        seed = set_seed(self.settings.module.seed_ship_ms_rail, self.logger, 'seed-ship-ms-rail')

        shipmentsRail = support.create_non_container_shipments_non_direct(
            self.tonnes.rail, self.terminals.rail, probabilities, self.distributionCenters,
            self.dcFractions, self.coeffsShip, self.coeffsDistDecay, self.skim,
            self.costFigures.vehicleType, self.costFigures.nonContainer,
            self.truckCapacities, self.dictBGtoNRM, self.zoneSettings.maximum_dutch,
            self.cumProbsCommodityToNSTR, self.dimVT, self.dimSS,
            self.dimAccessEgressGG, self.dimAccessGG,
            seed, self.moduleSettings.ship_discr_fac_ms_rail)

        self.logger.debug(
            f'\t\t{len(shipmentsRail)} spoorterminalzendingen niet-container aangemaakt.')

        self.logger.debug(
            "\tAanmakingen zendingen niet-gecontaineriseerd vervoer - binnenvaartterminals...")

        seed = set_seed(self.settings.module.seed_ship_ms_water, self.logger, 'seed-ship-ms-water')

        shipmentsWater = support.create_non_container_shipments_non_direct(
            self.tonnes.water, self.terminals.water, probabilities, self.distributionCenters,
            self.dcFractions, self.coeffsShip, self.coeffsDistDecay, self.skim,
            self.costFigures.vehicleType, self.costFigures.nonContainer,
            self.truckCapacities, self.dictBGtoNRM, self.zoneSettings.maximum_dutch,
            self.cumProbsCommodityToNSTR, self.dimVT, self.dimSS,
            self.dimAccessEgressGG, self.dimAccessGG,
            seed, self.moduleSettings.ship_discr_fac_ms_water)

        self.logger.debug(
            f'\t\t{len(shipmentsWater)} binnenvaartterminalzendingen niet-container aangemaakt.')

        return shipmentsDirect, shipmentsRail, shipmentsWater
