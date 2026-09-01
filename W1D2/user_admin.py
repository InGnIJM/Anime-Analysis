class User:
    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name

    def describe_user(self):
        print("Name: " + self.first_name + " " + self.last_name)

    def greet_user(self):
        print("Hello, " + self.first_name + " " + self.last_name + "!")


class Admin(User):
    def __init__(self, first_name, last_name, privileges):
        super().__init__(first_name, last_name)
        self.privileges = privileges

    def show_privileges(self):
        print("Privileges:")
        for privilege in self.privileges:
            print(privilege)


if __name__ == "__main__":
    admin = Admin(
        "Grace",
        "Hopper",
        ["can add post", "can delete post", "can ban user"],
    )
    admin.describe_user()
    admin.greet_user()
    admin.show_privileges()
