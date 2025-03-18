function convertToTimestamp(dateString) {
    // Extract parts from the input string
    let day = dateString.substring(0, 2);
    let month = dateString.substring(2, 4) - 1; // Months are zero-based in JS
    let year = dateString.substring(4, 8);
    let hours = dateString.substring(8, 10);
    let minutes = dateString.substring(10, 12);

    // Create a Date object
    let date = new Date(year, month, day, hours, minutes);

    // Return timestamp as a number (milliseconds since epoch)
    return date.getTime();
}

// Example usage
let timestamp = convertToTimestamp("190320251230"); // 19th March 2025, 12:30 PM
console.log(timestamp);

function convertFromTimestamp(timestamp) {
    let date = new Date(timestamp);

    let day = String(date.getDate()).padStart(2, '0');
    let month = String(date.getMonth() + 1).padStart(2, '0'); // Months are 0-based
    let year = date.getFullYear();
    let hours = String(date.getHours()).padStart(2, '0');
    let minutes = String(date.getMinutes()).padStart(2, '0');

    return `${day}${month}${year}${hours}${minutes}`;
}

// Example usage
let originalString = convertFromTimestamp(1742368200000); // Example timestamp
console.log(originalString); // Output: "190320251230"



let timestamp1 = convertToTimestamp("190320251230"); // 19th March 2025, 12:30 PM
let timestamp2 = convertToTimestamp("200320251200"); // 20th March 2025, 12:00 PM

console.log(timestamp1 < timestamp2); // true (19th March is earlier than 20th March)
console.log(timestamp1 > timestamp2); // false


