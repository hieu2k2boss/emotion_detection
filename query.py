import sqlite3
conn = sqlite3.connect('C:\\Users\\hieu2\\OneDrive\\Máy tính\\NLP\\Github\\emotion_detection\\chatbot_api\\cskh.db')

# Xóa theo thứ tự để tránh lỗi foreign key
conn.execute('DELETE FROM emotion_logs WHERE ticket_id BETWEEN 41 AND 59')
conn.execute('DELETE FROM messages    WHERE ticket_id BETWEEN 41 AND 59')
conn.execute('DELETE FROM tickets     WHERE id        BETWEEN 41 AND 59')

conn.commit()

# Kiểm tra lại
remaining = conn.execute('SELECT id FROM tickets ORDER BY id').fetchall()
print('Tickets còn lại:', [r[0] for r in remaining])
conn.close()