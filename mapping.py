import json


class MappingTrackId:

    def __init__(self, track_id, track_uri):
        self.track_id = track_id
        self.track_uri = track_uri

    def __str__(self):
        return {
            self.track_id: self.track_uri
        }

    def save_track(self):

        with open('trained-model/id_track_mapping.json', 'r+') as file:
            data = json.load(file)
            data['mapping'].append({
                self.track_id: self.track_uri
            })
            file.seek(0)
            json.dump(data, file, indent=4)
            file.truncate()