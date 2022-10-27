import logging
from typing import TypeVar

from settings import Settings

import calculation.tour.module as tour
import calculation.parcel_dmnd.module as parcel_dmnd
import calculation.parcel_schd.module as parcel_schd
import calculation.service.module as service
import calculation.traf.module as traf

from calculation.common.base_module import BaseModule
from calculation.ship.module import ShipModule
from calculation.output.module import OutputModule

logger = logging.getLogger(f"lwm.{__name__}")


TModule = TypeVar("TModule", bound=BaseModule)


class Core:
    def __init__(self, settings: Settings):
        logger.info("Initializing!")
        self.settings = settings

    def try_execute(self, func, logger):
        try:
            func(self.settings, logger)
        except BaseException as e:
            logger.exception("Module execution failed!")
            raise e

    def try_execute_module(self, module_class: TModule):
        try:
            module: BaseModule = module_class(self.settings, logger)
            module.run()
        except BaseException as e:
            logger.exception("Module execution failed!")
            raise e

    def execute(self):

        if self.settings.module.run_ship:
            logger.info("Running Shipment Synthesizer...")
            self.try_execute_module(ShipModule)
            logger.info("\tShipment Synthesizer finished.")

        if self.settings.module.run_tour:
            logger.info("Running Tour Formation...")
            self.try_execute(tour.run_module, logger)
            logger.info("\tTour Formation finished.")

        if self.settings.module.run_parcel_dmnd:
            logger.info("Running Parcel Demand...")
            self.try_execute(parcel_dmnd.run_module, logger)
            logger.info("\tParcel Demand finished.")

        if self.settings.module.run_parcel_schd:
            logger.info("Running Parcel Scheduling...")
            self.try_execute(parcel_schd.run_module, logger)
            logger.info("\tParcel Scheduling finished.")

        if self.settings.module.run_service:
            logger.info("Running Service / Construction..")
            self.try_execute(service.run_module, logger)
            logger.info("\tService / Construction finished.")

        if self.settings.module.run_traf:
            logger.info("Running Traffic Assignment..")
            self.try_execute(traf.run_module, logger)
            logger.info("\tTraffic Assignment finished.")

        if self.settings.module.run_output:
            logger.info("Running Output Module...")
            self.try_execute_module(OutputModule)
            logger.info("\tOutput Module finished.")
