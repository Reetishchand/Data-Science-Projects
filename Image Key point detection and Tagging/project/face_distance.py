import face_recognition

known_image_one = face_recognition.load_image_file("obama.jpg")
known_image_two = face_recognition.load_image_file("biden.jpg")

# Get the face encodings for the known images
face_encoding_one = face_recognition.face_encodings(known_image_one)[0]
face_encoding_two = face_recognition.face_encodings(known_image_two)[0]

known_encodings = [
    face_encoding_one,
    face_encoding_two
]

# Load a test image and get encondings for it
image_to_test = face_recognition.load_image_file("obama2.jpg")
image_to_test_encoding = face_recognition.face_encodings(image_to_test)[0]

# See how far apart the test image is from the known faces
face_distances = face_recognition.face_distance(known_encodings, image_to_test_encoding)

for i, face_distance in enumerate(face_distances):
    print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i))
    print("- With a normal cutoff of 0.6, would the test image match the known image? {}".format(face_distance < 0.6))
    print("- With a very strict cutoff of 0.5, would the test image match the known image? {}".format(face_distance < 0.5))
    print()
