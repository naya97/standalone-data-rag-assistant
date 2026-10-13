-- Active: 1790623941564@@127.0.0.1@3306
SELECT original_filename, chunk_index, length(chunk_text) AS text_length
FROM rag_chunks
ORDER BY text_length DESC
LIMIT 5;

SELECT original_filename, chunk_index, chunk_text 
FROM rag_chunks 
WHERE chunk_text LIKE '%\uf0b7%' 
   OR chunk_text LIKE '%\n \n%'
   OR chunk_text LIKE '%  %';


SELECT chunk_index, chunk_text 
FROM rag_chunks 
WHERE original_filename = 'Task-3.pdf' AND chunk_index IN (0, 1);


SELECT original_filename, section_header, COUNT(*) as chunks_count
FROM rag_chunks
WHERE section_header IS NOT NULL
GROUP BY original_filename, section_header;