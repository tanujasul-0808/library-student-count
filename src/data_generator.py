import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_library_data(start_date='2024-01-01', days=730):
    """
    Generates synthetic data for library usage.
    """
    date_range = pd.date_range(start=start_date, periods=days)
    data = []

    for date in date_range:
        # Basic Features
        day_of_week = date.dayofweek # 0=Monday, 6=Sunday
        month = date.month
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # 1. Is it a holiday? (Random ~5% chance + weekends are essentially "off" for classes)
        # We assume library is OPEN on weekends but classes are off
        is_holiday = 1 if (random.random() < 0.05) else 0 
        
        # 2. Is it Exam Week? (Assuming ~4 exam weeks per year)
        # Simple logic: If month is May(5) or Dec(12) and day is 10-20
        is_exam_week = 1 if (month in [5, 12] and 10 <= date.day <= 20) else 0

        # 3. Librarian Presence (Random ~90% attendance)
        is_librarian_present = 1 if random.random() < 0.9 else 0
        
        # 4. Total Students on Campus (Max 350)
        # Base: 300. Weekends: Drop to 50. Holiday: Drop to 20. Exam: Max out 340-350.
        if is_holiday:
            campus_count = random.randint(10, 50)
        elif is_weekend:
            campus_count = random.randint(30, 100)
        elif is_exam_week:
            campus_count = random.randint(320, 350)
        else:
            campus_count = random.randint(250, 330) # Normal days

        # 5. Target: Library Count
        # Logic: Base is % of campus count. 
        # Factors: Exam (+), Weekend (-), No Librarian (-), Holiday (-)
        
        if is_holiday and is_librarian_present == 0:
            library_count = 0 # Closed effectively
        else:
            base_rate = 0.3 # 30% of campus students usually go
            if is_exam_week: base_rate = 0.8 # 80% go during exams
            if is_weekend: base_rate = 0.1 # Few go on weekends
            
            # Librarian impact: If absent, count drops (cannot issue books)
            if is_librarian_present == 0: base_rate *= 0.5 
            
            library_count = int(campus_count * base_rate)
            
            # Add some noise/randomness
            noise = random.randint(-10, 10)
            library_count = max(0, min(library_count + noise, campus_count))

        data.append({
            'Date': date,
            'DayOfWeek': day_of_week,
            'IsHoliday': is_holiday,
            'IsExamWeek': is_exam_week,
            'IsLibrarianPresent': is_librarian_present,
            'TotalCampusStudents': campus_count,
            'LibraryStudentCount': library_count
        })

    return pd.DataFrame(data)

if __name__ == "__main__":
    # Test generation
    df = generate_library_data()
    print(df.head())