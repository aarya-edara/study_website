import os
import re
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score



# 1 = useful for revision
# 0 = usually not as useful for revision

# the starter data below is to be later replaced with more advanced notes to make the model more accurate/efficient

training_data = [

    # definitions

    ("Electric flux is the measure of the electric field passing through a surface.", 1),
    ("Voltage is the electric potential difference between two points.", 1),
    ("Current is the rate of flow of electric charge.", 1),
    ("Resistance is the opposition to the flow of electric current.", 1),
    ("Capacitance is the ability of a component to store electric charge.", 1),
    ("Inductance is the tendency of a conductor to oppose changes in current.", 1),
    ("Power is the rate at which energy is transferred.", 1),
    ("A vector has both magnitude and direction.", 1),
    ("A scalar quantity has magnitude but no direction.", 1),
    ("Divergence measures the net outward flow of a vector field.", 1),
    ("Curl measures the local rotation of a vector field.", 1),
    ("The gradient points in the direction of greatest increase of a scalar field.", 1),
    ("A diode is a semiconductor device that primarily allows current in one direction.", 1),
    ("A depletion region is an area around a pn junction depleted of mobile charge carriers.", 1),
    ("Diffusion current is caused by charge carriers moving from high concentration to low concentration.", 1),
    ("Drift current is caused by charge carriers moving under an electric field.", 1),
    ("Forward bias reduces the potential barrier of a pn junction.", 1),
    ("Reverse bias increases the width of the depletion region.", 1),
    ("A transistor is a semiconductor device used for switching or amplification.", 1),
    ("An operational amplifier is a high-gain differential voltage amplifier.", 1),

    # laws, equations and relationships

    ("Ohm's law states that voltage equals current multiplied by resistance.", 1),
    ("Gauss's law relates electric flux through a closed surface to the enclosed charge.", 1),
    ("Kirchhoff's current law states that the total current entering a node equals the total current leaving it.", 1),
    ("Kirchhoff's voltage law states that the sum of voltages around a closed loop is zero.", 1),
    ("Newton's second law states that force equals mass multiplied by acceleration.", 1),
    ("The electric field is force per unit positive test charge.", 1),
    ("Electrical power can be calculated using P equals VI.", 1),
    ("The time constant of an RC circuit is the product of resistance and capacitance.", 1),
    ("For capacitors in parallel, the equivalent capacitance is the sum of the individual capacitances.", 1),
    ("For resistors in series, the equivalent resistance is the sum of the individual resistances.", 1),
    ("The divergence of the curl of a vector field is zero.", 1),
    ("The curl of the gradient of a scalar field is zero.", 1),
    ("The dot product of perpendicular vectors is zero.", 1),
    ("The magnitude of a unit vector is one.", 1),

    # cause, effect + concepts

    ("Increasing reverse bias causes the depletion region to become wider.", 1),
    ("Forward bias allows significant current to flow through a diode.", 1),
    ("Increasing resistance decreases current when voltage is held constant.", 1),
    ("An electric field exerts a force on charged particles.", 1),
    ("Negative feedback can stabilise the gain of an amplifier.", 1),
    ("Electrons are the majority carriers in n-type semiconductor material.", 1),
    ("Holes are the majority carriers in p-type semiconductor material.", 1),
    ("At equilibrium, drift current and diffusion current balance each other.", 1),
    ("A changing magnetic flux can induce an electromotive force.", 1),
    ("The direction of the gradient is normal to a level surface.", 1),


    # methods and procedures

    ("To calculate electric flux, take the surface integral of the electric field dotted with the area vector.", 1),
    ("To determine whether vectors are linearly independent, solve the corresponding homogeneous equation.", 1),
    ("To calculate equivalent parallel resistance, add the reciprocals of the individual resistances.", 1),
    ("To find a directional derivative, take the dot product of the gradient with the unit direction vector.", 1),
    ("To apply Gauss's law, choose a Gaussian surface that exploits the symmetry of the charge distribution.", 1),
    ("The chain rule is used when differentiating a composition of functions.", 1),
    ("Partial differentiation treats the other independent variables as constants.", 1),

    # qualifications and exceptions

    ("An ideal diode has zero voltage drop when forward biased.", 1),
    ("A silicon diode is commonly approximated as having a forward voltage drop of about 0.7 volts.", 1),
    ("The electric field inside an ideal conductor in electrostatic equilibrium is zero.", 1),
    ("Gauss's law is always valid, although symmetry determines whether it is useful for calculating the field.", 1),
    ("A set containing the zero vector is linearly dependent.", 1),

    # content to be ignored - with score 0

    ("Welcome to today's lecture.", 0),
    ("Today we are going to continue our discussion.", 0),
    ("We discussed this briefly last week.", 0),
    ("We will return to this topic later.", 0),
    ("This slide contains the learning objectives.", 0),
    ("Please read chapter four before next week's lecture.", 0),
    ("You should already be familiar with this material.", 0),
    ("This is something we looked at previously.", 0),
    ("The next slide shows another example.", 0),
    ("Here is an interesting historical fact.", 0),
    ("Let's move on to the next section.", 0),
    ("We do not need to worry about this yet.", 0),
    ("This will become clearer later in the course.", 0),
    ("There are several things shown on this slide.", 0),
    ("You can find more information in the textbook.", 0),
    ("This topic will be covered in another lecture.", 0),
    ("Thank you for listening.", 0),
    ("Any questions before we continue?", 0),
    ("Let's take a short break.", 0),
    ("The lecturer will provide additional resources online.", 0),
    ("Remember to submit your coursework before the deadline.", 0),
    ("The assessment information is available on the module page.", 0),
    ("This example is left as an exercise for the reader.", 0),
    ("We have now reached the end of this lecture.", 0),

    # words/phrases that are not as useful or rich in content

    ("Example 1.", 0),
    ("Example 2.", 0),
    ("Lecture 3.", 0),
    ("Week four.", 0),
    ("Introduction.", 0),
    ("Summary.", 0),
    ("Contents.", 0),
    ("References.", 0),
    ("Figure 1.", 0),
    ("Table 2.", 0),
    ("Page 14.", 0),
    ("See below.", 0),
    ("As shown above.", 0),
    ("Consider the following.", 0),
]

# basic cleaning

def clean_sentence(sentence):
    sentence = str(sentence)

    # removing repeated whitespace
    sentence = re.sub(r"\s+", " ", sentence)

    return sentence.strip()


texts = [clean_sentence(text) for text, label in training_data]
labels = [label for text, label in training_data]


# train/test split

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)


# TF-IDF: Turns words/phrases into numerical features.

# Logistic regression: learns which features usually indicate useful revision sentences

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    )
])

# train

print("\nTraining revision importance model...\n")

model.fit(X_train, y_train)

# test

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("MODEL RESULTS")
print("=" * 50)
print(f"Accuracy: {accuracy:.3f}")
print()
print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Less important",
            "Important"
        ],
        zero_division=0
    )
)


# show examples

examples = [
    "Gauss's law states that electric flux depends on enclosed charge.",
    "Reverse bias causes the depletion region to increase in width.",
    "We will talk about this again next week.",
    "Please look at the diagram on the next slide.",
    "The gradient gives the direction of maximum increase."
]

print("\nEXAMPLE PREDICTIONS")
print("=" * 50)

for sentence in examples:

    probability = model.predict_proba([sentence])[0][1]

    print(
        f"{probability:.3f}  "
        f"{'IMPORTANT' if probability >= 0.5 else 'LESS IMPORTANT'}"
        f"  |  {sentence}"
    )

# Save model

model_directory = "models"

os.makedirs(
    model_directory,
    exist_ok=True
)

model_path = os.path.join(
    model_directory,
    "revision_model.pkl"
)

joblib.dump(
    model,
    model_path
)

print("\nModel saved successfully:")
print(model_path)
