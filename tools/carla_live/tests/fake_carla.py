"""Minimal stand-in for the CARLA Python API (0.9.15 subset used by the live viewer).

A straight two-lane road along UE +X, a building, a traffic light, an ego vehicle with a camera and
one traffic vehicle ahead. Vehicles drive along +X at constant speed when the world ticks.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from frames import ue_transform_matrix  # noqa: E402


class Vector3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)


Location = Vector3D


class Rotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch, self.yaw, self.roll = float(pitch), float(yaw), float(roll)


class Transform:
    def __init__(self, location=None, rotation=None):
        self.location = location or Location()
        self.rotation = rotation or Rotation()

    def get_matrix(self):
        l, r = self.location, self.rotation
        return ue_transform_matrix((l.x, l.y, l.z), (r.pitch, r.yaw, r.roll)).tolist()

    def get_right_vector(self):
        m = np.array(self.get_matrix())
        return Vector3D(*m[:3, 1])

    def get_forward_vector(self):
        m = np.array(self.get_matrix())
        return Vector3D(*m[:3, 0])


class BoundingBox:
    def __init__(self, location, extent, rotation=None):
        self.location, self.extent, self.rotation = location, extent, rotation or Rotation()


class Label:
    def __init__(self, name):
        self.name = name


class LaneMarking:
    def __init__(self, type_, color):
        self.type, self.color = Label(type_), Label(color)


class Waypoint:
    def __init__(self, x, y, lane_id, road_id=1, lane_width=3.5, s=0.0):
        self.transform = Transform(Location(x, y, 0.0), Rotation(yaw=0.0))
        self.lane_width, self.lane_id, self.road_id, self.s = lane_width, lane_id, road_id, s
        self.lane_type = Label("Driving")
        outer = LaneMarking("Solid", "White" if lane_id > 0 else "Yellow")
        inner = LaneMarking("Broken", "White")
        self.right_lane_marking = outer if lane_id > 0 else inner
        self.left_lane_marking = inner if lane_id > 0 else outer

    def next(self, d):
        if self.transform.location.x + d > 200.0:
            return []
        return [Waypoint(self.transform.location.x + d, self.transform.location.y, self.lane_id, self.road_id,
                         self.lane_width, self.s + d)]


class Map:
    name = "Carla/Maps/FakeTown"

    def generate_waypoints(self, d):
        wps = []
        x = 0.0
        while x <= 200.0:
            wps.append(Waypoint(x, 1.75, 1, s=x))
            wps.append(Waypoint(x, -1.75, -1, s=x))
            x += d
        return wps

    def get_spawn_points(self):
        return [Transform(Location(10.0, 1.75, 0.3))]


class Attr:
    def __init__(self, v):
        self.v = v

    def as_int(self):
        return int(self.v)


class Blueprint:
    def __init__(self, id_, attrs=None):
        self.id = id_
        self.attributes = dict(attrs or {})

    def set_attribute(self, k, v):
        self.attributes[k] = v

    def get_attribute(self, k):
        return Attr(self.attributes[k])


class BlueprintLibrary:
    def __init__(self):
        self.bps = [Blueprint("vehicle.lincoln.mkz_2020", {"number_of_wheels": 4, "color": "20,20,200"}),
                    Blueprint("vehicle.audi.a2", {"number_of_wheels": 4, "color": "200,10,10"}),
                    Blueprint("sensor.camera.rgb", {})]

    def filter(self, pattern):
        p = pattern.replace("*", "")
        return [Blueprint(b.id, b.attributes) for b in self.bps if b.id.startswith(p)]

    def find(self, id_):
        return next(Blueprint(b.id, b.attributes) for b in self.bps if b.id == id_)


class Actor:
    _next = 100

    def __init__(self, world, bp, transform, parent=None):
        Actor._next += 1
        self.id = Actor._next
        self.world, self.type_id, self.attributes = world, bp.id, dict(bp.attributes)
        self.rel = transform
        self.parent = parent
        self.speed = 0.0
        self.bounding_box = BoundingBox(Location(0, 0, 0.75), Vector3D(2.3, 1.0, 0.75))
        self.semantic_tags = [14] if self.type_id.startswith("vehicle.") else []
        self.light_state = 0
        self._cb = None

    def get_transform(self):
        if self.parent is None:
            return self.rel
        p = self.parent.get_transform().location
        r = self.rel.location
        return Transform(Location(p.x + r.x, p.y + r.y, p.z + r.z), self.rel.rotation)

    def set_autopilot(self, on, port=None):
        self.speed = 5.0 if on else 0.0

    def get_light_state(self):
        return self.light_state

    def listen(self, cb):
        self._cb = cb

    def stop(self):
        self._cb = None

    def destroy(self):
        if self in self.world.actors:
            self.world.actors.remove(self)


class TrafficLight(Actor):
    def __init__(self, world):
        super().__init__(world, Blueprint("traffic.traffic_light"), Transform(Location(40, 6, 0)))
        self.state = Label("Red")

    def get_state(self):
        return self.state

    def get_light_boxes(self):
        return [BoundingBox(Location(40, 5.5, 4.5), Vector3D(0.2, 0.2, 0.6))]


class EnvObject:
    def __init__(self, id_, type_, bb):
        self.id, self.type, self.bounding_box = id_, Label(type_), bb


class Image:
    def __init__(self, w, h, frame):
        self.width, self.height, self.frame = w, h, frame
        self.raw_data = bytes(w * h * 4)


class Settings:
    synchronous_mode = False
    fixed_delta_seconds = None


class Weather:
    sun_altitude_angle = 45.0
    sun_azimuth_angle = 30.0


class World:
    def __init__(self):
        self.actors = [TrafficLight(self)]
        self.frame = 0
        self.settings = Settings()
        self.weather = Weather()

    def get_map(self):
        return Map()

    def get_settings(self):
        return self.settings

    def apply_settings(self, s):
        self.settings = s

    def get_blueprint_library(self):
        return BlueprintLibrary()

    def try_spawn_actor(self, bp, tf):
        a = Actor(self, bp, tf)
        self.actors.append(a)
        if len([x for x in self.actors if x.type_id.startswith("vehicle.")]) == 1:
            # First vehicle is the ego; place one traffic vehicle 25 m ahead in the same lane.
            lead = Actor(self, Blueprint("vehicle.audi.a2", {"color": "200,10,10"}),
                         Transform(Location(tf.location.x + 25.0, tf.location.y, 0.0)))
            lead.light_state = 0x2  # low beam
            self.actors.append(lead)
        return a

    def spawn_actor(self, bp, tf, attach_to=None):
        a = Actor(self, bp, tf, parent=attach_to)
        self.actors.append(a)
        return a

    def get_actors(self):
        return list(self.actors)

    def get_weather(self):
        return self.weather

    def get_environment_objects(self, label=None):
        return [EnvObject(1, "Buildings", BoundingBox(Location(35, -12, 5), Vector3D(8, 4, 5))),
                EnvObject(2, "Roads", BoundingBox(Location(100, 0, 0), Vector3D(100, 4, 0.1)))]

    def tick(self):
        self.frame += 1
        dt = self.settings.fixed_delta_seconds or 0.05
        for a in self.actors:
            if a.parent is None and a.speed:
                a.rel = Transform(Location(a.rel.location.x + a.speed * dt, a.rel.location.y, a.rel.location.z),
                                  a.rel.rotation)
        for a in self.actors:
            if a._cb:
                a._cb(Image(int(a.attributes.get("image_size_x", 64)), int(a.attributes.get("image_size_y", 36)),
                            self.frame))
        return self.frame


class TrafficManager:
    def set_synchronous_mode(self, on):
        pass

    def get_port(self):
        return 8000


class Client:
    def __init__(self, *a, **k):
        self.world = World()

    def set_timeout(self, t):
        pass

    def get_world(self):
        return self.world

    def load_world(self, name):
        return self.world

    def get_trafficmanager(self, port=8000):
        return TrafficManager()
