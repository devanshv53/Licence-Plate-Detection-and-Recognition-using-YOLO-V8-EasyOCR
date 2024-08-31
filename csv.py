import csv
import datetime
# Initialize CSV file with headers if it doesn't exist
csv_file = 'vehicle_entry_exit_log.csv'

# Check if the file exists, otherwise create it and add headers
try:
    with open(csv_file, 'x', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['License Plate', 'Entry Time', 'Exit Time'])
except FileExistsError:
    pass

def log_vehicle(license_plate, entry=True):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        
        if entry:
            writer.writerow([license_plate, current_time, ''])
        else:
            # To update exit time for the same vehicle
            rows = []
            with open(csv_file, 'r') as read_file:
                reader = csv.reader(read_file)
                for row in reader:
                    if row[0] == license_plate and row[2] == '':
                        row[2] = current_time  # Update exit time
                    rows.append(row)
            
            # Write updated data back to the file
            with open(csv_file, 'w', newline='') as write_file:
                writer = csv.writer(write_file)
                writer.writerows(rows)

detected_plate = "ABC1234"  # Replace with your actual detection code

# Log the entry
log_vehicle(detected_plate, entry=True)

# Later, when the vehicle exits, log the exit
log_vehicle(detected_plate, entry=False)

detected_plate = "ABC1234"  # Replace with your actual detection code

# Log the entry
log_vehicle(detected_plate, entry=True)

# Later, when the vehicle exits, log the exit
log_vehicle(detected_plate, entry=False)
