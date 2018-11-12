import os
import glob
import random


class PairGenerator(object):
    person1 = 'person1'
    person2 = 'person2'
    label = 'same_person'

    def __init__(self, lfw_path='./tf_dataset/resources' + os.path.sep + 'lfw'):
        self.all_people = self.generate_all_people_dict(lfw_path)

    def generate_all_people_dict(self, lfw_path):
        # generates a dictionary between a person and all the photos of that person
        all_people = {}
        for person_folder in os.listdir(lfw_path):
            person_photos = glob.glob(lfw_path + os.path.sep + person_folder + os.path.sep + '*.jpg')
            all_people[person_folder] = person_photos
        return all_people

    def get_next_pair(self):
        all_people_names = list(self.all_people.keys())

        while True:
            # draw a person at random
            person1 = random.choice(all_people_names)
            # flip a coin to decide whether we fetch a photo of the same person vs different person

            same_person = random.random() > 0.5
            if same_person:
                person2 = person1
            else:
                # repeatedly pick random names until we find a different name
                person2 = person1
                while person2 == person1:
                    person2 = random.choice(all_people_names)

            person1_photo = random.choice(self.all_people[person1])
            person2_photo = random.choice(self.all_people[person2])
            yield ({self.person1: person1_photo,
                    self.person2: person2_photo,
                    self.label: same_person})
