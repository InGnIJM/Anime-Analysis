students = [
    {
        "student_id": "2023001",
        "name": "Zhang San",
        "class": "Software Engineering 1",
        "scores": {"Math": 95, "English": 88, "Java": 92},
    },
    {
        "student_id": "2023002",
        "name": "Li Si",
        "class": "Software Engineering 1",
        "scores": {"Math": 86, "English": 97, "Java": 84},
    },
    {
        "student_id": "2023003",
        "name": "Wang Wu",
        "class": "Software Engineering 1",
        "scores": {"Math": 90, "English": 85, "Java": 99},
    },
    {
        "student_id": "2023004",
        "name": "Zhao Liu",
        "class": "Software Engineering 1",
        "scores": {"Math": 93, "English": 94, "Java": 95},
    },
    {
        "student_id": "2023005",
        "name": "Chen Qi",
        "class": "Software Engineering 1",
        "scores": {"Math": 58, "English": 75, "Java": 52},
    },
    {
        "student_id": "2023006",
        "name": "Sun Ba",
        "class": "Software Engineering 1",
        "scores": {"Math": 75, "English": 55, "Java": 80},
    },
]

subjects = ["Math", "English", "Java"]


if __name__ == "__main__":
    for subject in subjects:
        first_student = students[0]
        for student in students:
            if student["scores"][subject] > first_student["scores"][subject]:
                first_student = student
        print(
            subject
            + " first: "
            + first_student["name"]
            + " "
            + str(first_student["scores"][subject])
        )

    total_first = students[0]
    total_first_score = sum(total_first["scores"].values())
    for student in students:
        total_score = sum(student["scores"].values())
        if total_score > total_first_score:
            total_first = student
            total_first_score = total_score
    print("Total first: " + total_first["name"] + " " + str(total_first_score))

    failed_students = []
    for student in students:
        for score in student["scores"].values():
            if score < 60:
                failed_students.append(student["name"])
                break

    print("Failed students:")
    for name in failed_students:
        print(name)
