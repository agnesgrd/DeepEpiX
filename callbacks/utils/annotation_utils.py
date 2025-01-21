from collections import Counter

def get_annotation_descriptions(annotations_store):
        """
        Extracts the list of unique annotation description names 
        from the annotations-store.

        Parameters:
        annotations_store (list): A list of dictionaries representing annotations.

        Returns:
        list: A list of unique description names.
        """
        if not annotations_store or not isinstance(annotations_store, list):
            return []

        # Extract descriptions
        descriptions = [annotation.get('description') for annotation in annotations_store if 'description' in annotation]

        # Count occurrences of each description
        description_counts = Counter(descriptions)

        # Return unique descriptions
        return description_counts