# Titanic Dataset Model Comparison

---

## Dataset

The dataset used in this project is the Titanic dataset, available on [Kaggle](https://www.kaggle.com/competitions/titanic). It includes the following key features:
- **PassengerId**: Unique identifier for each passenger.
- **Survived**: Survival status (0 = No, 1 = Yes).
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
- **Name**: Passenger's name.
- **Sex**: Passenger's gender.
- **Age**: Passenger's age.
- **SibSp**: Number of siblings/spouses aboard.
- **Parch**: Number of parents/children aboard.
- **Ticket**: Ticket number.
- **Fare**: Ticket fare.
- **Cabin**: Cabin number (if available).
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

Download the dataset directly from [Kaggle](https://www.kaggle.com/c/titanic) and place it in a `data/` directory in this repository.

---

## Preprocessing

### PassengerId and Ticket

**PassengerId** are dropped as they are not relevant to the study.

### Cabin

1. **Preprocess Cabin Information**:
   - Extract the first character from the `Cabin` column to create a new column `Deck`.
   - Generate additional features:
     - `Has_Cabin`: Binary indicator showing whether the cabin information is available.
     - `Deck_category`: Group decks into categories:
       - `'ABC'`: Upper decks
       - `'DE'`: Middle decks
       - `'FG'`: Lower decks
       - `'Other'`: Special cases
   - If `Pclass` and `Fare` are available:
     - Calculate the ratio of the fare paid to the average fare for the passenger class (`Fare_ratio`).
     - Estimate missing `Deck_category` values based on class and fare ratio.

### Name

   - Extract titles from the `Name` column using regex.
   - Standardize and group rare titles under the category `Rare`.
   - Harmonize similar titles (e.g., `Mlle` → `Miss`, `Mme` → `Mrs`).

### Age

   - Fill missing `Age` values by computing the median age for each combination of `Title` and `Pclass`.
   - If no matching median exists, fallback to median age by `Title` or the overall median age.

### SibSp and Parch

   - Create new features:
     - `FamilySize`: Total family members (sum of `SibSp` and `Parch`, plus one for the passenger).
     - `IsAlone`: Binary feature indicating whether the passenger is traveling alone.
     - `FarePerPerson`: Average fare per family member.

### Embarked

   - Fill missing `Embarked` values with the mode (`'S'` in this case).

### General Preprocessing Pipeline
   - Remove duplicate rows
   - Sequentially apply all preprocessing functions
   - Removed columns: 'PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Deck', 'Has_Cabin', 'IsAlone'
   - Numerical imputer: mean strategy
   - Categories encoding: OneHot

---

Feel free to run the code and see what model performs best!
