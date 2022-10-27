from __future__ import annotations

import logging
import os
import re
from functools import reduce
from typing import Dict, List

import yaml

logger = logging.getLogger(f"lwm.{__name__}")

IGNORED_FILE_KEYS = ["id", "code", "path"]


class Settings:
    """
    Reads YAML-settingsfile and makes settings available.
    """

    def __init__(self, settings_path: str):
        """
        Store the YAML-settingsfile content.

        Args:
            settings_path (str): Path of the YAML-settingsfile.
        """
        self._data = {}

        self._settings_dir = os.path.dirname(os.path.abspath(settings_path))

        with open(settings_path, 'r', encoding='utf8') as yamlfile:
            data = yaml.load(yamlfile, yaml.FullLoader)
            if not isinstance(data, dict):
                return
            self._data = data

    def get_path(self, id: str = None, code: str = None, **kwargs: str) -> str:
        """
        Get the path of a file from the YAML-settingsfile content,
        based on either an id or a file-code,
        as well as on zero or more key value pairs
        to get the path for a certain year, commodity etc.

        Args:
            id (str): Provided file id.
            code (str): Provided BasGoed filecode (see Code/Files.json for an example).
            kwargs (Dict[str, str]): Key value pairs to define a certain year, commodity etc.

        Returns:
            str: The path of the requested file.
        """
        def is_match(file: Dict[str, str]):
            if not (
                file.get("id", "") == id or file.get("code", "") == code
            ):
                return False
            return all(
                kwargs.get(key) == value
                for key, value in file.items()
                if key not in IGNORED_FILE_KEYS and key in kwargs
            )

        paths = [
            file.get("path", None)
            for file in self._data.get("files", [])
            if is_match(file)]

        if not paths or len(paths) > 1:
            raise FileNotFoundError(f"Geen uniek bestand voor '{id or code}' gevonden!")

        path = paths[0]

        path = reduce(
            lambda prev, cur: re.sub(fr"\[{cur[0].replace('_', '-')}\]", cur[1], prev),
            list(kwargs.items()), path)

        path = os.sep.join(re.split(r'[/\\]', path))

        if not os.path.isabs(path):
            path = os.path.join(self._settings_dir, path)

        return path

    @property
    def commodities(self) -> List[int]:
        """
        Get the commodities that need calculation.

        Returns:
            List[int]: A list of id's of commodities.
        """
        return self._data.get("commodities", [])

    @property
    def zones(self) -> ZoneSettings:
        """
        Get a ZoneSettings instance, based on the YAML-settingsfile content,
        holding information on zones.

        Returns:
            ZoneSettings: A ZoneSettings instance.
        """
        return ZoneSettings(self._data)

    @property
    def dimensions(self) -> DimensionSettings:
        """
        Get a DimensionSettings instance, based on the YAML-settingsfile content,
        holding element id's of certain dimensions.

        Returns:
            DimensionSettings: A DimensionSettings instance.
        """
        return DimensionSettings(self._data)

    @property
    def module(self) -> ModuleSettings:
        """
        Get a ModuleSettings instance, based on the YAML-settingsfile content,
        holding settings specific for this module,

        Returns:
            ModuleSettings: A ModuleSettings instance.
        """
        return ModuleSettings(self._data)


class ZoneSettings:
    """
    Makes zone settings in the YAML-settingsfile available.
    """

    def __init__(self, data: dict):
        """
        Store the provided dictionary from the YAML-settingsfile content,
        containing zone settings information.

        Args:
            data (dict): A dictionary containing the YAML-settingsfile content.
        """
        self._data = data.get("zones", {})

    @property
    def minimum_dutch(self) -> int:
        """
        Get the lowest Dutch zone id.

        Returns:
            int: The lowest Dutch zone id.
        """
        return self._data.get("minimum-dutch", 0)

    @property
    def maximum_dutch(self) -> int:
        """
        Get the highest Dutch zone id.

        Returns:
            int: The highest Dutch zone id.
        """
        return self._data.get("maximum-dutch", 0)

    @property
    def maximum_dutch_ck(self) -> int:
        """
        Get the highest Dutch zone id (CK zones).

        Returns:
            int: The highest Dutch zone id (CK-zones).
        """
        return self._data.get("maximum-dutch-ck", 0)

    @property
    def maximum_dutch_nrm(self) -> int:
        """
        Get the highest Dutch zone id (NRM zones).

        Returns:
            int: The highest Dutch zone id (NRM-zones).
        """
        return self._data.get("maximum-dutch-nrm", 0)

    @property
    def minimum_european(self) -> int:
        """
        Get the lowest European zone id.

        Returns:
            int: The lowest European zone id.
        """
        return self._data.get("minimum-european", 0)

    @property
    def maximum_european(self) -> int:
        """
        Get the highest European zone id.

        Returns:
            int: The highest European zone id.
        """
        return self._data.get("maximum-european", 0)

    @property
    def minimum_world(self) -> int:
        """
        Get the lowest 'overseas' zone id.

        Returns:
            int: The lowest 'overseas' zone id.
        """
        return self._data.get("minimum-world", 0)

    @property
    def maximum_world(self) -> int:
        """
        Get the highest 'overseas' zone id.

        Returns:
            int: The highest 'overseas' zone id.
        """
        return self._data.get("maximum-world", 0)

    @property
    def harbour_zones(self) -> List[int]:
        """
        Get the id's of all the harbour zones.

        Returns:
            List[int]: A list of id's of all the harbour zones.
        """
        return self._data.get("harbour-zones", [])


class DimensionSettings:
    """
    Makes dimension settings in the YAML-settingsfile available.
    """

    def __init__(self, data: dict):
        """
        Store the provided dictionary from the YAML-settingsfile content,
        containing dimension settings information.

        Args:
            data (dict): A dictionary containing the YAML-settingsfile content.
        """
        self._data = data.get("dimensions", {})

    @property
    def year(self) -> List[str]:
        """
        Get the id's of all years.

        Returns:
            List[str]: A list of id's of all years.
        """
        return self._data.get("year", [])

    @property
    def container(self) -> List[str]:
        """
        Get the id's of all items in the container dimension.

        Returns:
            List[str]: A list of id's of all items in the container dimension.
        """
        return self._data.get("container", [])

    @property
    def mode(self) -> List[str]:
        """
        Get the id's of all transport modes.

        Returns:
            List[str]: A list of id's of all transport modes.
        """
        return self._data.get("mode", [])

    @property
    def vehicle_type(self) -> List[str]:
        """
        Get the id's of all vehicle types.

        Returns:
            List[str]: A list of id's of all vehicle types.
        """
        return self._data.get("vehicle-type", [])

    @property
    def time_of_day(self) -> List[str]:
        """
        Get the id's of all items in the time-of-day dimension.

        Returns:
            List[str]: A list of id's of all items in the time-of-day dimension.
        """
        return [
            str(item)
            for item in self._data.get("time-of-day", [])]

    @property
    def combustion_type(self) -> List[str]:
        """
        Get the id's of all items in the combustion-type dimension.

        Returns:
            List[str]: A list of id's of all items in the combustion-type dimension.
        """
        return [
            str(item)
            for item in self._data.get("combustion-type", [])]

    @property
    def emission_type(self) -> List[str]:
        """
        Get the id's of all items in the emission-type dimension.

        Returns:
            List[str]: A list of id's of all items in the emission-type dimension.
        """
        return [
            str(item)
            for item in self._data.get("emission-type", [])]

    @property
    def nstr(self) -> List[str]:
        """
        Get the id's of all items in the nstr dimension.

        Returns:
            List[str]: A list of id's of all items in the nstr dimension.
        """
        return [
            str(item)
            for item in self._data.get("nstr", [])]

    @property
    def logistic_segment(self) -> List[str]:
        """
        Get the id's of all items in the logistic-segment dimension.

        Returns:
            List[str]: A list of id's of all items in the logistic-segment dimension.
        """
        return [
            str(item)
            for item in self._data.get("logistic-segment", [])]

    @property
    def road_type(self) -> List[str]:
        """
        Get the id's of all items in the road-type dimension.

        Returns:
            List[str]: A list of id's of all items in the road-type dimension.
        """
        return [
            str(item)
            for item in self._data.get("road-type", [])]

    @property
    def shipment_size(self) -> List[str]:
        """
        Get the id's of all items in the shipment-size dimension.

        Returns:
            List[str]: A list of id's of all items in the shipment-size dimension.
        """
        return [
            item
            for item in self._data.get("shipment-size", [])]

    @property
    def van_segment(self) -> List[str]:
        """
        Get the id's of all items in the van-segment dimension.

        Returns:
            List[str]: A list of id's of all items in the van-segment dimension.
        """
        return [
            item
            for item in self._data.get("van-segment", [])]

    @property
    def vehicle_type_lwm(self) -> List[str]:
        """
        Get the id's of all vehicle types.

        Returns:
            List[str]: A list of id's in the vehicle-type-lwm dimension.
        """
        return [
            item
            for item in self._data.get("vehicle-type-lwm", [])]

    @property
    def employment_category(self) -> List[str]:
        """
        Get the id's of all items in the employment-category dimension.

        Returns:
            List[str]: A list of id's of all items in the employment-category dimension.
        """
        return [
            str(item)
            for item in self._data.get("employment-category", [])]

    @property
    def commodities_access_egress(self) -> List[str]:
        """
        Get the id's of all items in the commodities-access-egress dimension.

        Returns:
            List[str]: A list of id's of all items in the commodities-access-egress dimension.
        """
        return [
            str(item)
            for item in self._data.get("commodities-access-egress", [])]

    @property
    def commodities_access(self) -> List[str]:
        """
        Get the id's of all items in the commodities-access dimension.

        Returns:
            List[str]: A list of id's of all items in the commodities-access dimension dimension.
        """
        return [
            str(item)
            for item in self._data.get("commodities-access", [])]

    @property
    def length_class(self) -> List[str]:
        """
        Get the id's of all items in the length-class dimension.

        Returns:
            List[str]: A list of id's of all items in the length-class dimension.
        """
        return [
            str(item)
            for item in self._data.get("length-class", [])]


class ModuleSettings:
    """
    Makes module specific settings in the YAML-settingsfile available.
    """

    def __init__(self, data: dict):
        """
        Store the provided dictionary from the YAML-settingsfile content,
        containing module specific settings information.

        Args:
            data (dict): A dictionary containing the YAML-settingsfile content.
        """
        self._data = data.get("lwm-module", {})

    @property
    def year_factor_goods(self) -> float:
        """
        Get the 'year-factor-goods' value from the YAML-settingsfile content.

        Returns:
            float: The 'year-factor-goods' value.
        """
        return self._data.get("year-factor-goods", None)

    @property
    def year_factor_service(self) -> float:
        """
        Get the 'year-factor-service' value from the YAML-settingsfile content.

        Returns:
            float: The 'year-factor-service' value.
        """
        return self._data.get("year-factor-service", None)

    @property
    def year_factor_construction(self) -> float:
        """
        Get the 'year-factor-construction' value from the YAML-settingsfile content.

        Returns:
            float: The 'year-factor-construction' value.
        """
        return self._data.get("year-factor-construction", None)

    @property
    def year_factor_parcel(self) -> float:
        """
        Get the 'year-factor-parcel' value from the YAML-settingsfile content.

        Returns:
            float: The 'year-factor-parcel' value.
        """
        return self._data.get("year-factor-parcel", None)

    @property
    def run_ship(self) -> bool:
        """
        Get the 'run-ship' value from the YAML-settingsfile content.

        Returns:
            float: The 'run-ship' value.
        """
        return self._data.get("run-ship", None)

    @property
    def run_tour(self) -> bool:
        """
        Get the 'run-tour' value from the YAML-settingsfile content.

        Returns:
            float: The 'run-tour' value.
        """
        return self._data.get("run-tour", None)

    @property
    def run_parcel_dmnd(self) -> bool:
        """
        Get the 'run-parcel-dmnd' value from the YAML-settingsfile content.

        Returns:
            float: The 'run-parcel-dmnd' value.
        """
        return self._data.get("run-parcel-dmnd", None)

    @property
    def run_parcel_schd(self) -> bool:
        """
        Get the 'run-parcel-schd' value from the YAML-settingsfile content.

        Returns:
            float: The 'run-parcel-schd' value.
        """
        return self._data.get("run-parcel-schd", None)

    @property
    def run_service(self) -> bool:
        """
        Get the 'run-service' value from the YAML-settingsfile content.

        Returns:
            float: The 'run-service' value.
        """
        return self._data.get("run-service", None)

    @property
    def run_traf(self) -> bool:
        """
        Get the 'run-traf' value from the YAML-settingsfile content.

        Returns:
            float: The 'run-traf' value.
        """
        return self._data.get("run-traf", None)

    @property
    def run_output(self) -> bool:
        """
        Get the 'run-output' value from the YAML-settingsfile content.

        Returns:
            float: The 'run-output' value.
        """
        return self._data.get("run-output", None)

    @property
    def seed_ship_ck(self) -> float:
        """
        Get the 'seed-ship-ck' value from the YAML-settingsfile content.

        Returns:
            float: The 'seed-ship-ck' value.
        """
        return self._data.get("seed-ship-ck", None)

    @property
    def seed_ship_ms_direct(self) -> float:
        """
        Get the 'seed-ms-direct' value from the YAML-settingsfile content.

        Returns:
            float: The 'seed-ship-ms-direct' value.
        """
        return self._data.get("seed-ship-ms-direct", None)

    @property
    def seed_ship_ms_rail(self) -> float:
        """
        Get the 'seed-ms-rail' value from the YAML-settingsfile content.

        Returns:
            float: The 'seed-ship-ms-rail' value.
        """
        return self._data.get("seed-ship-ms-rail", None)

    @property
    def seed_ship_ms_water(self) -> float:
        """
        Get the 'seed-ms-water' value from the YAML-settingsfile content.

        Returns:
            float: The 'seed-ship-ms-water' value.
        """
        return self._data.get("seed-ship-ms-water", None)

    @property
    def seed_tour_assign_carriers(self) -> float:
        """
        Get the 'seed-tour-assign-carriers' value from the YAML-settingsfile content.

        Returns:
            float: The 'seed-tour-assign-carriers' value.
        """
        return self._data.get("seed-tour-assign-carriers", None)

    @property
    def seed_tour_formation(self) -> float:
        """
        Get the 'seed-tour-formation' value from the YAML-settingsfile content.

        Returns:
            float: The 'seed-tour-formation' value.
        """
        return self._data.get("seed-tour-formation", None)

    @property
    def seed_tour_departure_times(self) -> float:
        """
        Get the 'seed-tour-departure-times' value from the YAML-settingsfile content.

        Returns:
            float: The 'seed-tour-assign-departure-times' value.
        """
        return self._data.get("seed-tour-assign-departure-times", None)

    @property
    def seed_tour_combustion_types(self) -> float:
        """
        Get the 'seed-tour-combustion-types' value from the YAML-settingsfile content.

        Returns:
            float: The 'seed-tour-combustion-types' value.
        """
        return self._data.get("seed-tour-combustion-types", None)

    @property
    def seed_parcel_dmnd(self) -> float:
        """
        Get the 'seed-parcel-dmnd' value from the YAML-settingsfile content.

        Returns:
            float: The 'seed-parcel-dmnd' value.
        """
        return self._data.get("seed-parcel-dmnd", None)

    @property
    def seed_parcel_schd(self) -> float:
        """
        Get the 'seed-parcel-schd' value from the YAML-settingsfile content.

        Returns:
            float: The 'seed-parcel-schd' value.
        """
        return self._data.get("seed-parcel-schd", None)

    @property
    def write_shape_ship(self) -> bool:
        """
        Get the 'write-shape-ship' value from the YAML-settingsfile content.

        Returns:
            bool: The 'write-shape-ship' value.
        """
        return self._data.get("write-shape-ship", None)

    @property
    def write_shape_tour(self) -> bool:
        """
        Get the 'write-shape-tour' value from the YAML-settingsfile content.

        Returns:
            bool: The 'write-shape-tour' value.
        """
        return self._data.get("write-shape-tour", None)

    @property
    def write_shape_parcel_dmnd(self) -> bool:
        """
        Get the 'write-shape-parcel-dmnd' value from the YAML-settingsfile content.

        Returns:
            bool: The 'write-shape-parcel-dmnd' value.
        """
        return self._data.get("write-shape-parcel-dmnd", None)

    @property
    def write_shape_parcel_schd(self) -> bool:
        """
        Get the 'write-shape-parcel-schd' value from the YAML-settingsfile content.

        Returns:
            bool: The 'write-shape-parcel-schd' value.
        """
        return self._data.get("write-shape-parcel-schd", None)

    @property
    def write_shape_traf(self) -> bool:
        """
        Get the 'write-shape-traf' value from the YAML-settingsfile content.

        Returns:
            bool: The 'write-shape-traf' value.
        """
        return self._data.get("write-shape-traf", None)

    @property
    def apply_zez(self) -> bool:
        """
        Get the 'apply-zez' value from the YAML-settingsfile content.

        Returns:
            bool: The 'apply-zez' value.
        """
        return self._data.get("apply-zez", None)

    @property
    def empl_per_m2_dc(self) -> float:
        """
        Get the 'empl-per-m2-dc' value from the YAML-settingsfile content.

        Returns:
            float: The 'empl-per-m2-dc' value.
        """
        return self._data.get("empl-per-m2-dc", None)

    @property
    def ship_discr_fac_ck(self) -> float:
        """
        Get the 'ship-discr-fac-ck' value from the YAML-settingsfile content.

        Returns:
            float: The 'ship-discr-fac-ck' value.
        """
        return self._data.get("ship-discr-fac-ck", None)

    @property
    def ship_discr_fac_ms_direct(self) -> float:
        """
        Get the 'ship-discr-fac-ms-direct' value from the YAML-settingsfile content.

        Returns:
            float: The 'ship-discr-fac-ms-direct' value.
        """
        return self._data.get("ship-discr-fac-ms-direct", None)

    @property
    def ship_discr_fac_ms_rail(self) -> float:
        """
        Get the 'ship-discr-fac-ms-rail' value from the YAML-settingsfile content.

        Returns:
            float: The 'ship-discr-fac-ms-rail' value.
        """
        return self._data.get("ship-discr-fac-ms-rail", None)

    @property
    def ship_discr_fac_ms_water(self) -> float:
        """
        Get the 'ship-discr-fac-ms-water' value from the YAML-settingsfile content.

        Returns:
            float: The 'ship-discr-fac-ms-water' value.
        """
        return self._data.get("ship-discr-fac-ms-water", None)

    @property
    def tour_num_cpu(self) -> float:
        """
        Get the 'tour-num-cpu' value from the YAML-settingsfile content.

        Returns:
            float: The 'tour-num-cpu' value.
        """
        return self._data.get("tour-num-cpu", None)

    @property
    def tour_max_num_ships(self) -> float:
        """
        Get the 'tour-max-num-ships' value from the YAML-settingsfile content.

        Returns:
            float: The 'tour-max-num-ships' value.
        """
        return self._data.get("tour-max-num-ships", None)

    @property
    def tour_max_empty_trip_distance(self) -> float:
        """
        Get the 'tour-max-empty-trip-distance' value from the YAML-settingsfile content.

        Returns:
            float: The 'tour-max-empty-trip-distance' value.
        """
        return self._data.get("tour-max-empty-trip-distance", None)

    @property
    def tour_num_carriers_non_dc(self) -> float:
        """
        Get the 'tour-num-carriers-non-dc' value from the YAML-settingsfile content.

        Returns:
            float: The 'tour-num-carriers-non-dc' value.
        """
        return self._data.get("tour-num-carriers-non-dc", None)

    @property
    def service_tolerance(self) -> float:
        """
        Get the 'service-tolerance' value from the YAML-settingsfile content.

        Returns:
            float: The 'service-tolerance' value.
        """
        return self._data.get("service-tolerance", None)

    @property
    def service_max_num_iter(self) -> float:
        """
        Get the 'service-max-num-iter' value from the YAML-settingsfile content.

        Returns:
            float: The 'service-max-num-iter' value.
        """
        return self._data.get("service-max-num-iter", None)

    @property
    def traf_num_cpu(self) -> float:
        """
        Get the 'traf-num-cpu' value from the YAML-settingsfile content.

        Returns:
            float: The 'traf-num-cpu' value.
        """
        return self._data.get("traf-num-cpu", None)

    @property
    def traf_num_multiroute(self) -> float:
        """
        Get the 'traf-num-multiroute' value from the YAML-settingsfile content.

        Returns:
            float: The 'traf-num-multiroute' value.
        """
        return self._data.get("traf-num-multiroute", None)
