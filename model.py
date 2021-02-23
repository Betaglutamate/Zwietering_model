import analysisFunctions as af

class Plate:

    def __init__(self, name, osmolyte, temperature, date, raw_data):
        self.name = name
        self.osmolyte = osmolyte
        self.temperature = temperature
        self.date = date
        self.raw_data = raw_data

    def clean_data(self):
        self.clean_data = af.analyze_plate(self.raw_data)
        return self.clean_data


    