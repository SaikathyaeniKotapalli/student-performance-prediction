
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load sample student data
data = pd.read_csv("student_data.csv")

# Select inputs and output
X = data[
    [
        "study_hours_per_day",
        "attendance_percentage",
        "previous_marks",
    ]
]
y = data["passed"]

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Check model accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model accuracy: {accuracy:.2%}")
print("This is a learning project using sample data.")

# Get student details
hours = float(input("Study hours per day: "))
attendance = float(input("Attendance percentage: "))
marks = float(input("Previous marks: "))

if not (0 <= hours <= 24 and 0 <= attendance <= 100
        and 0 <= marks <= 100):
    print("Please enter valid values.")
else:
    student = pd.DataFrame([{
        "study_hours_per_day": hours,
        "attendance_percentage": attendance,
        "previous_marks": marks
    }])

    result = model.predict(student)[0]

    if result == 1:
        print("Model result: Likely to pass.")
    else:
        print("Model result: May need more support.")
