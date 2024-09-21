data = {
    "G43109": [
        {
            "description": "Migraine with aura, not intractable, w/o status migrainosus",
            "full_description": "Migraine with aura, not intractable, without status migrainosus",
            "disease_name": "Migraine with aura, not intractable"
        }
    ],
    "G43111": [
        {
            "description": "Migraine with aura, intractable, with status migrainosus",
            "full_description": "Migraine with aura, intractable, with status migrainosus",
            "disease_name": "Migraine with aura, intractable"
        }
    ],
    "G43119": [
        {
            "description": "Migraine with aura, intractable, without status migrainosus",
            "full_description": "Migraine with aura, intractable, without status migrainosus",
            "disease_name": "Migraine with aura, intractable"
        }
    ]
}

# Convert the dictionary to a list of tuples

location=[1,3]

# based on the location extract only those elements from the dictionary,but the keys and values are not similar to location

# Expected Output:
# [('G43111', [{'description': 'Migraine with aura, intractable, with status migrainosus', 'full_description': 'Migraine with aura, intractable, with status migrainosus', 'disease_name': 'Migraine with aura, intractable'}])]

# Explanation:
# The location has 1 and 3, so the output is the elements present at the 1 and 3 index in the dictionary.

# The first element in the dictionary is at index 0, the second element in the dictionary is at index 1, the third element in the dictionary is at index 2, and so on.

