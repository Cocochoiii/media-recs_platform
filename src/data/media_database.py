"""
Rich Media Database for Recommendation System

Contains 100+ movies/shows with real metadata for demonstration.
"""

import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class Genre(Enum):
    ACTION = "Action"
    COMEDY = "Comedy"
    DRAMA = "Drama"
    SCIFI = "Sci-Fi"
    HORROR = "Horror"
    ROMANCE = "Romance"
    THRILLER = "Thriller"
    ANIMATION = "Animation"
    DOCUMENTARY = "Documentary"
    FANTASY = "Fantasy"
    CRIME = "Crime"
    MYSTERY = "Mystery"
    ADVENTURE = "Adventure"
    FAMILY = "Family"
    MUSIC = "Music"


@dataclass
class MediaItem:
    id: int
    title: str
    year: int
    genres: List[str]
    rating: float
    poster: str
    backdrop: str
    description: str
    duration: int  # minutes
    director: str
    cast: List[str]
    popularity: float
    media_type: str  # movie, tv, music


# Real movie database with TMDB-style poster URLs
MEDIA_DATABASE: List[Dict[str, Any]] = [
    # ===== ACTION =====
    {
        "id": 1,
        "title": "The Dark Knight",
        "year": 2008,
        "genres": ["Action", "Crime", "Drama"],
        "rating": 9.0,
        "poster": "https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/nMKdUUepR0i5zn0y1T4CsSB5chy.jpg",
        "description": "When the menace known as the Joker wreaks havoc on Gotham, Batman must confront one of the greatest tests of his ability to fight injustice.",
        "duration": 152,
        "director": "Christopher Nolan",
        "cast": ["Christian Bale", "Heath Ledger", "Aaron Eckhart"],
        "popularity": 95.5,
        "media_type": "movie"
    },
    {
        "id": 2,
        "title": "Inception",
        "year": 2010,
        "genres": ["Action", "Sci-Fi", "Thriller"],
        "rating": 8.8,
        "poster": "https://image.tmdb.org/t/p/w500/edv5CZvWj09upOsy2Y6IwDhK8bt.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/s3TBrRGB1iav7gFOCNx3H31MoES.jpg",
        "description": "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.",
        "duration": 148,
        "director": "Christopher Nolan",
        "cast": ["Leonardo DiCaprio", "Joseph Gordon-Levitt", "Elliot Page"],
        "popularity": 94.2,
        "media_type": "movie"
    },
    {
        "id": 3,
        "title": "Mad Max: Fury Road",
        "year": 2015,
        "genres": ["Action", "Adventure", "Sci-Fi"],
        "rating": 8.1,
        "poster": "https://image.tmdb.org/t/p/w500/8tZYtuWezp8JbcsvHYO0O46tFbo.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/phszHPFVhPHhMZgo0fWTKBDQsJA.jpg",
        "description": "In a post-apocalyptic wasteland, a woman rebels against a tyrannical ruler in search for her homeland.",
        "duration": 120,
        "director": "George Miller",
        "cast": ["Tom Hardy", "Charlize Theron", "Nicholas Hoult"],
        "popularity": 88.7,
        "media_type": "movie"
    },
    {
        "id": 4,
        "title": "John Wick",
        "year": 2014,
        "genres": ["Action", "Thriller", "Crime"],
        "rating": 7.4,
        "poster": "https://image.tmdb.org/t/p/w500/fZPSd91yGE9fCcCe6OoQr6E3Bev.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/umC04Cozevu8nn3JTDJ1pc7PVTn.jpg",
        "description": "An ex-hitman comes out of retirement to track down the gangsters that killed his dog and took his car.",
        "duration": 101,
        "director": "Chad Stahelski",
        "cast": ["Keanu Reeves", "Michael Nyqvist", "Alfie Allen"],
        "popularity": 86.3,
        "media_type": "movie"
    },
    {
        "id": 5,
        "title": "Top Gun: Maverick",
        "year": 2022,
        "genres": ["Action", "Drama"],
        "rating": 8.3,
        "poster": "https://image.tmdb.org/t/p/w500/62HCnUTziyWcpDaBO2i1DX17ljH.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/AaV1YIdWKhxX9W2YYGgVnMHuFGv.jpg",
        "description": "After thirty years, Maverick is still pushing the envelope as a top naval aviator.",
        "duration": 130,
        "director": "Joseph Kosinski",
        "cast": ["Tom Cruise", "Miles Teller", "Jennifer Connelly"],
        "popularity": 92.1,
        "media_type": "movie"
    },

    # ===== SCI-FI =====
    {
        "id": 6,
        "title": "Interstellar",
        "year": 2014,
        "genres": ["Sci-Fi", "Adventure", "Drama"],
        "rating": 8.6,
        "poster": "https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/xJHokMbljvjADYdit5fK5VQsXEG.jpg",
        "description": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.",
        "duration": 169,
        "director": "Christopher Nolan",
        "cast": ["Matthew McConaughey", "Anne Hathaway", "Jessica Chastain"],
        "popularity": 93.8,
        "media_type": "movie"
    },
    {
        "id": 7,
        "title": "Blade Runner 2049",
        "year": 2017,
        "genres": ["Sci-Fi", "Drama", "Mystery"],
        "rating": 8.0,
        "poster": "https://image.tmdb.org/t/p/w500/gajva2L0rPYkEWjzgFlBXCAVBE5.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/ilRyazdMJwN05exqhwK4tMKBYZs.jpg",
        "description": "Young Blade Runner K's discovery of a long-buried secret leads him to track down former Blade Runner Rick Deckard.",
        "duration": 164,
        "director": "Denis Villeneuve",
        "cast": ["Ryan Gosling", "Harrison Ford", "Ana de Armas"],
        "popularity": 85.4,
        "media_type": "movie"
    },
    {
        "id": 8,
        "title": "Dune",
        "year": 2021,
        "genres": ["Sci-Fi", "Adventure", "Drama"],
        "rating": 8.0,
        "poster": "https://image.tmdb.org/t/p/w500/d5NXSklXo0qyIYkgV94XAgMIckC.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/jYEW5xZkZk2WTrdbMGAPFuBqbDc.jpg",
        "description": "Paul Atreides, a brilliant and gifted young man, must travel to the most dangerous planet in the universe.",
        "duration": 155,
        "director": "Denis Villeneuve",
        "cast": ["Timothée Chalamet", "Rebecca Ferguson", "Zendaya"],
        "popularity": 91.2,
        "media_type": "movie"
    },
    {
        "id": 9,
        "title": "The Matrix",
        "year": 1999,
        "genres": ["Sci-Fi", "Action"],
        "rating": 8.7,
        "poster": "https://image.tmdb.org/t/p/w500/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/fNG7i7RqMErkcqhohV2a6cV1Ehy.jpg",
        "description": "A computer hacker learns about the true nature of reality and his role in the war against its controllers.",
        "duration": 136,
        "director": "Lana Wachowski",
        "cast": ["Keanu Reeves", "Laurence Fishburne", "Carrie-Anne Moss"],
        "popularity": 89.5,
        "media_type": "movie"
    },
    {
        "id": 10,
        "title": "Avatar",
        "year": 2009,
        "genres": ["Sci-Fi", "Adventure", "Fantasy"],
        "rating": 7.9,
        "poster": "https://image.tmdb.org/t/p/w500/jRXYjXNq0Cs2TcJjLkki24MLp7u.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/o0s4XsEDfDlvit5pDRKjzXR4pp2.jpg",
        "description": "A paraplegic Marine dispatched to Pandora on a unique mission becomes torn between following his orders.",
        "duration": 162,
        "director": "James Cameron",
        "cast": ["Sam Worthington", "Zoe Saldana", "Sigourney Weaver"],
        "popularity": 87.3,
        "media_type": "movie"
    },

    # ===== DRAMA =====
    {
        "id": 11,
        "title": "The Shawshank Redemption",
        "year": 1994,
        "genres": ["Drama", "Crime"],
        "rating": 9.3,
        "poster": "https://image.tmdb.org/t/p/w500/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/kXfqcdQKsToO0OUXHcrrNCHDBzO.jpg",
        "description": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
        "duration": 142,
        "director": "Frank Darabont",
        "cast": ["Tim Robbins", "Morgan Freeman", "Bob Gunton"],
        "popularity": 96.8,
        "media_type": "movie"
    },
    {
        "id": 12,
        "title": "Forrest Gump",
        "year": 1994,
        "genres": ["Drama", "Romance"],
        "rating": 8.8,
        "poster": "https://image.tmdb.org/t/p/w500/arw2vcBveWOVZr6pxd9XTd1TdQa.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/qdIMHd4sEfJSckfVJfKQvisL02a.jpg",
        "description": "The presidencies of Kennedy and Johnson through the eyes of an Alabama man with an IQ of 75.",
        "duration": 142,
        "director": "Robert Zemeckis",
        "cast": ["Tom Hanks", "Robin Wright", "Gary Sinise"],
        "popularity": 91.5,
        "media_type": "movie"
    },
    {
        "id": 13,
        "title": "The Godfather",
        "year": 1972,
        "genres": ["Drama", "Crime"],
        "rating": 9.2,
        "poster": "https://image.tmdb.org/t/p/w500/3bhkrj58Vtu7enYsRolD1fZdja1.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/tmU7GeKVybMWFButWEGl2M4GeiP.jpg",
        "description": "The aging patriarch of an organized crime dynasty transfers control of his empire to his reluctant son.",
        "duration": 175,
        "director": "Francis Ford Coppola",
        "cast": ["Marlon Brando", "Al Pacino", "James Caan"],
        "popularity": 94.2,
        "media_type": "movie"
    },
    {
        "id": 14,
        "title": "Schindler's List",
        "year": 1993,
        "genres": ["Drama", "History"],
        "rating": 9.0,
        "poster": "https://image.tmdb.org/t/p/w500/sF1U4EUQS8YHUYjNl3pMGNIQyr0.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/loRmRzQXZeqG78TqZuyvSlEQfZb.jpg",
        "description": "In German-occupied Poland, industrialist Oskar Schindler becomes concerned for his Jewish workforce.",
        "duration": 195,
        "director": "Steven Spielberg",
        "cast": ["Liam Neeson", "Ben Kingsley", "Ralph Fiennes"],
        "popularity": 88.9,
        "media_type": "movie"
    },
    {
        "id": 15,
        "title": "Parasite",
        "year": 2019,
        "genres": ["Drama", "Thriller", "Comedy"],
        "rating": 8.5,
        "poster": "https://image.tmdb.org/t/p/w500/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/TU9NIjwzjoKPwQHoHshkFcQUCG.jpg",
        "description": "Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family.",
        "duration": 132,
        "director": "Bong Joon-ho",
        "cast": ["Song Kang-ho", "Lee Sun-kyun", "Cho Yeo-jeong"],
        "popularity": 90.3,
        "media_type": "movie"
    },

    # ===== THRILLER =====
    {
        "id": 16,
        "title": "Se7en",
        "year": 1995,
        "genres": ["Thriller", "Crime", "Mystery"],
        "rating": 8.6,
        "poster": "https://image.tmdb.org/t/p/w500/6yoghtyTpznpBik8EngEmJskVUO.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/sKCr78MXSLixwmZ8DyJLrpMsd15.jpg",
        "description": "Two detectives hunt a serial killer who uses the seven deadly sins as his motives.",
        "duration": 127,
        "director": "David Fincher",
        "cast": ["Brad Pitt", "Morgan Freeman", "Gwyneth Paltrow"],
        "popularity": 87.6,
        "media_type": "movie"
    },
    {
        "id": 17,
        "title": "The Silence of the Lambs",
        "year": 1991,
        "genres": ["Thriller", "Crime", "Horror"],
        "rating": 8.6,
        "poster": "https://image.tmdb.org/t/p/w500/uS9m8OBk1A8eM9I042bx8XXpqAq.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/mfwq2nMBzArzQ7Y9RKE8SKeeTkg.jpg",
        "description": "A young FBI cadet must receive the help of an incarcerated cannibal killer to catch another serial killer.",
        "duration": 118,
        "director": "Jonathan Demme",
        "cast": ["Jodie Foster", "Anthony Hopkins", "Scott Glenn"],
        "popularity": 86.4,
        "media_type": "movie"
    },
    {
        "id": 18,
        "title": "Gone Girl",
        "year": 2014,
        "genres": ["Thriller", "Drama", "Mystery"],
        "rating": 8.1,
        "poster": "https://image.tmdb.org/t/p/w500/lv5xShBIDPe7m5pLKIlfdzfcQUq.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/yCT9C9qfZqMmFRiLapQjivxPOIT.jpg",
        "description": "A man becomes the main suspect in his wife's disappearance on their fifth wedding anniversary.",
        "duration": 149,
        "director": "David Fincher",
        "cast": ["Ben Affleck", "Rosamund Pike", "Neil Patrick Harris"],
        "popularity": 84.7,
        "media_type": "movie"
    },
    {
        "id": 19,
        "title": "Shutter Island",
        "year": 2010,
        "genres": ["Thriller", "Mystery", "Drama"],
        "rating": 8.2,
        "poster": "https://image.tmdb.org/t/p/w500/kve20tXwUZpu4GUX8l6X7Z4jmL6.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/94Ve6vsJnxllWyPwL3rQ4iJPeVT.jpg",
        "description": "A U.S. Marshal investigates the disappearance of a murderer who escaped from a hospital for the criminally insane.",
        "duration": 138,
        "director": "Martin Scorsese",
        "cast": ["Leonardo DiCaprio", "Emily Mortimer", "Mark Ruffalo"],
        "popularity": 85.9,
        "media_type": "movie"
    },
    {
        "id": 20,
        "title": "Prisoners",
        "year": 2013,
        "genres": ["Thriller", "Crime", "Drama"],
        "rating": 8.1,
        "poster": "https://image.tmdb.org/t/p/w500/uhOjHEK5ZobXfReyLdwwQB9pUor.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/xgDj56UWyeWQcxQ44f5A3RTWuSs.jpg",
        "description": "When his daughter and her friend go missing, a desperate father takes matters into his own hands.",
        "duration": 153,
        "director": "Denis Villeneuve",
        "cast": ["Hugh Jackman", "Jake Gyllenhaal", "Viola Davis"],
        "popularity": 83.8,
        "media_type": "movie"
    },

    # ===== HORROR =====
    {
        "id": 21,
        "title": "Get Out",
        "year": 2017,
        "genres": ["Horror", "Thriller", "Mystery"],
        "rating": 7.7,
        "poster": "https://image.tmdb.org/t/p/w500/tFXcEccSQMf3lfhfXKSU9iRBpa3.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/mXN4Gw9tZJVKrLJHde2IcUHmV3P.jpg",
        "description": "A young African-American visits his white girlfriend's parents for the weekend, where his simmering uneasiness leads to horrifying discovery.",
        "duration": 104,
        "director": "Jordan Peele",
        "cast": ["Daniel Kaluuya", "Allison Williams", "Bradley Whitford"],
        "popularity": 82.5,
        "media_type": "movie"
    },
    {
        "id": 22,
        "title": "A Quiet Place",
        "year": 2018,
        "genres": ["Horror", "Sci-Fi", "Drama"],
        "rating": 7.5,
        "poster": "https://image.tmdb.org/t/p/w500/nAU74GmpUk7t5iklEp3bufwDq4n.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/roYyPiQDQKmIKUEhO912693tSja.jpg",
        "description": "A family must live in silence to avoid mysterious creatures that hunt by sound.",
        "duration": 90,
        "director": "John Krasinski",
        "cast": ["Emily Blunt", "John Krasinski", "Millicent Simmonds"],
        "popularity": 80.3,
        "media_type": "movie"
    },
    {
        "id": 23,
        "title": "Hereditary",
        "year": 2018,
        "genres": ["Horror", "Drama", "Mystery"],
        "rating": 7.3,
        "poster": "https://image.tmdb.org/t/p/w500/p9fmuz2Oz0uSiBtcgME8IXYFNBC.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/hFvL8dFcQlEtHGmAr0O98O6zH4z.jpg",
        "description": "A grieving family is haunted by tragic and disturbing occurrences after the death of their secretive grandmother.",
        "duration": 127,
        "director": "Ari Aster",
        "cast": ["Toni Collette", "Milly Shapiro", "Gabriel Byrne"],
        "popularity": 78.9,
        "media_type": "movie"
    },
    {
        "id": 24,
        "title": "The Conjuring",
        "year": 2013,
        "genres": ["Horror", "Thriller", "Mystery"],
        "rating": 7.5,
        "poster": "https://image.tmdb.org/t/p/w500/wVYREutTvI2tmxr6ujrHT704wGF.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/ng6atggWIGEME93xthfwTmF8VU0.jpg",
        "description": "Paranormal investigators work to help a family terrorized by a dark presence in their farmhouse.",
        "duration": 112,
        "director": "James Wan",
        "cast": ["Vera Farmiga", "Patrick Wilson", "Lili Taylor"],
        "popularity": 81.2,
        "media_type": "movie"
    },
    {
        "id": 25,
        "title": "It",
        "year": 2017,
        "genres": ["Horror", "Fantasy"],
        "rating": 7.3,
        "poster": "https://image.tmdb.org/t/p/w500/9E2y5Q7WlCVNEhP5GiVTjhEhx1o.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/tcheoA2nPATCm2vvXw2hVQoaEFD.jpg",
        "description": "A group of bullied kids band together when a monster, taking the appearance of a clown, begins hunting children.",
        "duration": 135,
        "director": "Andy Muschietti",
        "cast": ["Bill Skarsgård", "Jaeden Martell", "Finn Wolfhard"],
        "popularity": 83.6,
        "media_type": "movie"
    },

    # ===== COMEDY =====
    {
        "id": 26,
        "title": "The Grand Budapest Hotel",
        "year": 2014,
        "genres": ["Comedy", "Drama", "Adventure"],
        "rating": 8.1,
        "poster": "https://image.tmdb.org/t/p/w500/eWdyYQreja6JGCzqHWXpWHDrrPo.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/nX5XotM9yprCKarRH4fzOq1VM1J.jpg",
        "description": "A writer encounters the owner of an aging high-class hotel, who tells him of his early years serving as a lobby boy.",
        "duration": 99,
        "director": "Wes Anderson",
        "cast": ["Ralph Fiennes", "F. Murray Abraham", "Mathieu Amalric"],
        "popularity": 84.1,
        "media_type": "movie"
    },
    {
        "id": 27,
        "title": "Superbad",
        "year": 2007,
        "genres": ["Comedy"],
        "rating": 7.6,
        "poster": "https://image.tmdb.org/t/p/w500/ek8e8txUyUwd2BNqj6lFEerJfbq.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/lp7WXZmKAr3hLdEf5OrLFtfKHqH.jpg",
        "description": "Two co-dependent high school seniors are forced to deal with separation anxiety after their plan to become party legends fails.",
        "duration": 113,
        "director": "Greg Mottola",
        "cast": ["Jonah Hill", "Michael Cera", "Christopher Mintz-Plasse"],
        "popularity": 75.8,
        "media_type": "movie"
    },
    {
        "id": 28,
        "title": "The Hangover",
        "year": 2009,
        "genres": ["Comedy"],
        "rating": 7.7,
        "poster": "https://image.tmdb.org/t/p/w500/uluhlXubGu1VxU63X9VHCLWDAYP.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/wPBJxPJwTweOHfNxI5btZ7KQSLA.jpg",
        "description": "Three buddies wake up from a bachelor party in Las Vegas with no memory and the bachelor missing.",
        "duration": 100,
        "director": "Todd Phillips",
        "cast": ["Bradley Cooper", "Ed Helms", "Zach Galifianakis"],
        "popularity": 79.4,
        "media_type": "movie"
    },
    {
        "id": 29,
        "title": "Knives Out",
        "year": 2019,
        "genres": ["Comedy", "Crime", "Drama"],
        "rating": 7.9,
        "poster": "https://image.tmdb.org/t/p/w500/pThyQovXQrw2m0s9x82twj48Jq4.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/4HWAQu28e2yaWrtupFPGFkdNU7V.jpg",
        "description": "A detective investigates the death of a patriarch of an eccentric, combative family.",
        "duration": 130,
        "director": "Rian Johnson",
        "cast": ["Daniel Craig", "Chris Evans", "Ana de Armas"],
        "popularity": 85.2,
        "media_type": "movie"
    },
    {
        "id": 30,
        "title": "Jojo Rabbit",
        "year": 2019,
        "genres": ["Comedy", "Drama", "War"],
        "rating": 7.9,
        "poster": "https://image.tmdb.org/t/p/w500/7GsM4mtM0worCtIVeiQt28HieeN.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/agoBZfL1q5G79SD0npArSlJn8BH.jpg",
        "description": "A young German boy in the Hitler Youth whose imaginary friend is Hitler discovers his mother is hiding a Jewish girl in their home.",
        "duration": 108,
        "director": "Taika Waititi",
        "cast": ["Roman Griffin Davis", "Thomasin McKenzie", "Scarlett Johansson"],
        "popularity": 82.7,
        "media_type": "movie"
    },

    # ===== ANIMATION =====
    {
        "id": 31,
        "title": "Spider-Man: Into the Spider-Verse",
        "year": 2018,
        "genres": ["Animation", "Action", "Adventure"],
        "rating": 8.4,
        "poster": "https://image.tmdb.org/t/p/w500/iiZZdoQBEYBv6id8su7ImL0oCbD.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/7d6EY00g1c39SGZOoCJ5Py9nNth.jpg",
        "description": "Teen Miles Morales becomes Spider-Man and joins forces with his heroes in an epic battle.",
        "duration": 117,
        "director": "Bob Persichetti",
        "cast": ["Shameik Moore", "Jake Johnson", "Hailee Steinfeld"],
        "popularity": 88.9,
        "media_type": "movie"
    },
    {
        "id": 32,
        "title": "Coco",
        "year": 2017,
        "genres": ["Animation", "Family", "Fantasy"],
        "rating": 8.4,
        "poster": "https://image.tmdb.org/t/p/w500/gGEsBPAijhVUFoiNpgZXqRVWJt2.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/askg3SMvhqEl4OL52YuvdtY40Yb.jpg",
        "description": "Aspiring musician Miguel enters the Land of the Dead to find his great-great-grandfather, a legendary singer.",
        "duration": 105,
        "director": "Lee Unkrich",
        "cast": ["Anthony Gonzalez", "Gael García Bernal", "Benjamin Bratt"],
        "popularity": 87.5,
        "media_type": "movie"
    },
    {
        "id": 33,
        "title": "Your Name",
        "year": 2016,
        "genres": ["Animation", "Romance", "Fantasy"],
        "rating": 8.4,
        "poster": "https://image.tmdb.org/t/p/w500/q719jXXEzOoYaps6babgKnONONX.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/dIWwZW7dJJtqC6CgWzYkNVKIUm8.jpg",
        "description": "Two teenagers share a profound, magical connection upon discovering they are swapping bodies.",
        "duration": 106,
        "director": "Makoto Shinkai",
        "cast": ["Ryunosuke Kamiki", "Mone Kamishiraishi"],
        "popularity": 86.3,
        "media_type": "movie"
    },
    {
        "id": 34,
        "title": "Spirited Away",
        "year": 2001,
        "genres": ["Animation", "Family", "Fantasy"],
        "rating": 8.6,
        "poster": "https://image.tmdb.org/t/p/w500/39wmItIWsg5sZMyRUHLkWBcuVCM.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/mSDsSDwaP3E7dEfUPWy4J0djt4O.jpg",
        "description": "During her family's move to the suburbs, a sullen 10-year-old wanders into a world ruled by gods and witches.",
        "duration": 125,
        "director": "Hayao Miyazaki",
        "cast": ["Rumi Hiiragi", "Miyu Irino", "Mari Natsuki"],
        "popularity": 89.7,
        "media_type": "movie"
    },
    {
        "id": 35,
        "title": "Toy Story 4",
        "year": 2019,
        "genres": ["Animation", "Adventure", "Comedy"],
        "rating": 7.7,
        "poster": "https://image.tmdb.org/t/p/w500/w9kR8qbmQ01HwnvK4alvnQ2ca0L.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/m67smI1IIMmYzCl9axvKNULVKLr.jpg",
        "description": "When a new toy called Forky joins Woody and the gang, a road trip alongside old and new friends reveals how big the world can be.",
        "duration": 100,
        "director": "Josh Cooley",
        "cast": ["Tom Hanks", "Tim Allen", "Annie Potts"],
        "popularity": 83.2,
        "media_type": "movie"
    },

    # ===== ROMANCE =====
    {
        "id": 36,
        "title": "La La Land",
        "year": 2016,
        "genres": ["Romance", "Drama", "Music"],
        "rating": 8.0,
        "poster": "https://image.tmdb.org/t/p/w500/uDO8zWDhfWwoFdKS4fzkUJt0Rf0.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/nadTlnTE6DdgmYsN4iWc2a2wiaI.jpg",
        "description": "A jazz pianist falls for an aspiring actress in Los Angeles while they pursue their dreams.",
        "duration": 128,
        "director": "Damien Chazelle",
        "cast": ["Ryan Gosling", "Emma Stone", "John Legend"],
        "popularity": 86.8,
        "media_type": "movie"
    },
    {
        "id": 37,
        "title": "The Notebook",
        "year": 2004,
        "genres": ["Romance", "Drama"],
        "rating": 7.8,
        "poster": "https://image.tmdb.org/t/p/w500/rNzQyW4f8B8cQeg7Dgj3n6eT5k9.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/qom1SZSENdmHFNZBXbtJAU0WTlC.jpg",
        "description": "A poor yet passionate young man falls in love with a rich young woman, giving her a sense of freedom.",
        "duration": 123,
        "director": "Nick Cassavetes",
        "cast": ["Ryan Gosling", "Rachel McAdams", "James Garner"],
        "popularity": 82.4,
        "media_type": "movie"
    },
    {
        "id": 38,
        "title": "Pride & Prejudice",
        "year": 2005,
        "genres": ["Romance", "Drama"],
        "rating": 7.8,
        "poster": "https://image.tmdb.org/t/p/w500/nYqB07qvXnpfkMntVP90dALnzR0.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/xKzLsPWmE8CZpLHuxHXAfLxdKqJ.jpg",
        "description": "Sparks fly when spirited Elizabeth Bennet meets single, rich, and proud Mr. Darcy.",
        "duration": 129,
        "director": "Joe Wright",
        "cast": ["Keira Knightley", "Matthew Macfadyen", "Brenda Blethyn"],
        "popularity": 79.6,
        "media_type": "movie"
    },
    {
        "id": 39,
        "title": "Eternal Sunshine of the Spotless Mind",
        "year": 2004,
        "genres": ["Romance", "Drama", "Sci-Fi"],
        "rating": 8.3,
        "poster": "https://image.tmdb.org/t/p/w500/5MwkWH9tYHv3mV9OdYTMR5qreIz.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/7y3eYvTsGjxPcv6aNvPNvf8u9sh.jpg",
        "description": "When their relationship turns sour, a couple undergoes a medical procedure to have each other erased from their memories.",
        "duration": 108,
        "director": "Michel Gondry",
        "cast": ["Jim Carrey", "Kate Winslet", "Tom Wilkinson"],
        "popularity": 84.5,
        "media_type": "movie"
    },
    {
        "id": 40,
        "title": "Before Sunrise",
        "year": 1995,
        "genres": ["Romance", "Drama"],
        "rating": 8.1,
        "poster": "https://image.tmdb.org/t/p/w500/77sqmZcrrj6rxNHtaKfOhzwF9OL.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/nLMCGl0eR8VA2ZG5oTI93OoF3lh.jpg",
        "description": "A young man and woman meet on a train in Europe, and spend one evening together in Vienna.",
        "duration": 101,
        "director": "Richard Linklater",
        "cast": ["Ethan Hawke", "Julie Delpy"],
        "popularity": 77.8,
        "media_type": "movie"
    },

    # ===== TV SHOWS =====
    {
        "id": 41,
        "title": "Breaking Bad",
        "year": 2008,
        "genres": ["Drama", "Crime", "Thriller"],
        "rating": 9.5,
        "poster": "https://image.tmdb.org/t/p/w500/ggFHVNu6YYI5L9pCfOacjizRGt.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/tsRy63Mu5cu8etL1X7ZLyf7UFMQ.jpg",
        "description": "A high school chemistry teacher diagnosed with cancer turns to making meth to secure his family's future.",
        "duration": 49,
        "director": "Vince Gilligan",
        "cast": ["Bryan Cranston", "Aaron Paul", "Anna Gunn"],
        "popularity": 97.2,
        "media_type": "tv"
    },
    {
        "id": 42,
        "title": "Game of Thrones",
        "year": 2011,
        "genres": ["Drama", "Fantasy", "Adventure"],
        "rating": 9.2,
        "poster": "https://image.tmdb.org/t/p/w500/1XS1oqL89opfnbLl8WnZY1O1uJx.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/suopoADq0k8YZr4dQXcU6pToj6s.jpg",
        "description": "Nine noble families fight for control over the lands of Westeros, while an ancient enemy returns.",
        "duration": 57,
        "director": "David Benioff",
        "cast": ["Emilia Clarke", "Peter Dinklage", "Kit Harington"],
        "popularity": 96.8,
        "media_type": "tv"
    },
    {
        "id": 43,
        "title": "Stranger Things",
        "year": 2016,
        "genres": ["Drama", "Fantasy", "Horror"],
        "rating": 8.7,
        "poster": "https://image.tmdb.org/t/p/w500/49WJfeN0moxb9IPfGn8AIqMGskD.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/56v2KjBlU4XaOv9rVYEQypROD7P.jpg",
        "description": "When a young boy disappears, his mother, a police chief, and his friends must confront terrifying supernatural forces.",
        "duration": 51,
        "director": "The Duffer Brothers",
        "cast": ["Millie Bobby Brown", "Finn Wolfhard", "Winona Ryder"],
        "popularity": 94.5,
        "media_type": "tv"
    },
    {
        "id": 44,
        "title": "The Office",
        "year": 2005,
        "genres": ["Comedy"],
        "rating": 8.9,
        "poster": "https://image.tmdb.org/t/p/w500/qWnJzyZhyy74gjpSjIXWmuk0ifX.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/vNpuAxGTl9HsUbHqam3E9CzqCvX.jpg",
        "description": "A mockumentary on a group of typical office workers, where the workday consists of ego clashes and inappropriate behavior.",
        "duration": 22,
        "director": "Greg Daniels",
        "cast": ["Steve Carell", "Rainn Wilson", "John Krasinski"],
        "popularity": 91.3,
        "media_type": "tv"
    },
    {
        "id": 45,
        "title": "The Crown",
        "year": 2016,
        "genres": ["Drama", "History"],
        "rating": 8.6,
        "poster": "https://image.tmdb.org/t/p/w500/1M876KPjulVwppEpldhdc8V4o68.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/1PgBcuMujmOcbUSSeVmS2NYQyx6.jpg",
        "description": "Follows the political rivalries and romance of Queen Elizabeth II's reign and the events that shaped the second half of the twentieth century.",
        "duration": 58,
        "director": "Peter Morgan",
        "cast": ["Claire Foy", "Olivia Colman", "Imelda Staunton"],
        "popularity": 88.7,
        "media_type": "tv"
    },

    # ===== MORE MOVIES =====
    {
        "id": 46,
        "title": "The Avengers",
        "year": 2012,
        "genres": ["Action", "Adventure", "Sci-Fi"],
        "rating": 8.0,
        "poster": "https://image.tmdb.org/t/p/w500/RYMX2wcKCBAr24UyPD7xwmjaTn.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/kwUQFeFXOOpgloMgZaadhzkbTI4.jpg",
        "description": "Earth's mightiest heroes must come together to stop Loki and his alien army from enslaving humanity.",
        "duration": 143,
        "director": "Joss Whedon",
        "cast": ["Robert Downey Jr.", "Chris Evans", "Scarlett Johansson"],
        "popularity": 93.1,
        "media_type": "movie"
    },
    {
        "id": 47,
        "title": "Avengers: Endgame",
        "year": 2019,
        "genres": ["Action", "Adventure", "Drama"],
        "rating": 8.4,
        "poster": "https://image.tmdb.org/t/p/w500/or06FN3Dka5tukK1e9sl16pB3iy.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/7RyHsO4yDXtBv1zUU3mTpHeQ0d5.jpg",
        "description": "After devastating events, the Avengers assemble once more to reverse Thanos' actions and restore balance.",
        "duration": 181,
        "director": "Anthony Russo",
        "cast": ["Robert Downey Jr.", "Chris Evans", "Mark Ruffalo"],
        "popularity": 95.6,
        "media_type": "movie"
    },
    {
        "id": 48,
        "title": "Black Panther",
        "year": 2018,
        "genres": ["Action", "Adventure", "Sci-Fi"],
        "rating": 7.3,
        "poster": "https://image.tmdb.org/t/p/w500/uxzzxijgPIY7slzFvMotPv8wjKA.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/b6ZJZHUdMEFECvGiDpJjlfUWela.jpg",
        "description": "T'Challa returns home to Wakanda to take his rightful place as king after his father's death.",
        "duration": 134,
        "director": "Ryan Coogler",
        "cast": ["Chadwick Boseman", "Michael B. Jordan", "Lupita Nyong'o"],
        "popularity": 89.4,
        "media_type": "movie"
    },
    {
        "id": 49,
        "title": "Joker",
        "year": 2019,
        "genres": ["Crime", "Drama", "Thriller"],
        "rating": 8.4,
        "poster": "https://image.tmdb.org/t/p/w500/udDclJoHjfjb8Ekgsd4FDteOkCU.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/n6bUvigpRFqSwmPp1m2YMDNqKpc.jpg",
        "description": "A mentally troubled comedian embarks on a downward spiral that leads to the creation of an iconic villain.",
        "duration": 122,
        "director": "Todd Phillips",
        "cast": ["Joaquin Phoenix", "Robert De Niro", "Zazie Beetz"],
        "popularity": 92.8,
        "media_type": "movie"
    },
    {
        "id": 50,
        "title": "1917",
        "year": 2019,
        "genres": ["Drama", "War"],
        "rating": 8.3,
        "poster": "https://image.tmdb.org/t/p/w500/iZf0KyrE25z1sage4SYFLCCrMi9.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/tUWivz05fcY14K6RzicRm7IHkUD.jpg",
        "description": "Two British soldiers must cross enemy territory to deliver a message that will stop a deadly attack.",
        "duration": 119,
        "director": "Sam Mendes",
        "cast": ["George MacKay", "Dean-Charles Chapman", "Mark Strong"],
        "popularity": 87.9,
        "media_type": "movie"
    },

    # ===== MORE CONTENT =====
    {
        "id": 51,
        "title": "Whiplash",
        "year": 2014,
        "genres": ["Drama", "Music"],
        "rating": 8.5,
        "poster": "https://image.tmdb.org/t/p/w500/7fn624j5lj3xTme2SgiLCeuedmO.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/6bbZ6XyvgfjhQwbplnUh1LSj1ky.jpg",
        "description": "A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor who will stop at nothing.",
        "duration": 106,
        "director": "Damien Chazelle",
        "cast": ["Miles Teller", "J.K. Simmons", "Melissa Benoist"],
        "popularity": 86.2,
        "media_type": "movie"
    },
    {
        "id": 52,
        "title": "The Social Network",
        "year": 2010,
        "genres": ["Drama", "Biography"],
        "rating": 7.8,
        "poster": "https://image.tmdb.org/t/p/w500/n0ybibhJtQ5icDqTp8eRytcIHJx.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/cqL4VW0fdzI7iLOgbKxgEJGTZYk.jpg",
        "description": "The story of how Mark Zuckerberg created Facebook while studying at Harvard.",
        "duration": 120,
        "director": "David Fincher",
        "cast": ["Jesse Eisenberg", "Andrew Garfield", "Justin Timberlake"],
        "popularity": 82.5,
        "media_type": "movie"
    },
    {
        "id": 53,
        "title": "The Wolf of Wall Street",
        "year": 2013,
        "genres": ["Comedy", "Crime", "Drama"],
        "rating": 8.2,
        "poster": "https://image.tmdb.org/t/p/w500/34m2tygAYBGqA9MXKhRDtzYd4MR.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/8IkxZqduPaP3cXGk0HgEOzPLq3W.jpg",
        "description": "Based on the true story of Jordan Belfort, from his rise to a wealthy stock-broker to his fall involving crime and corruption.",
        "duration": 180,
        "director": "Martin Scorsese",
        "cast": ["Leonardo DiCaprio", "Jonah Hill", "Margot Robbie"],
        "popularity": 88.3,
        "media_type": "movie"
    },
    {
        "id": 54,
        "title": "Fight Club",
        "year": 1999,
        "genres": ["Drama", "Thriller"],
        "rating": 8.8,
        "poster": "https://image.tmdb.org/t/p/w500/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/hZkgoQYus5vegHoetLkCJzb17zJ.jpg",
        "description": "An insomniac office worker and a soap salesman build a global organization to help vent male aggression.",
        "duration": 139,
        "director": "David Fincher",
        "cast": ["Brad Pitt", "Edward Norton", "Helena Bonham Carter"],
        "popularity": 90.1,
        "media_type": "movie"
    },
    {
        "id": 55,
        "title": "Pulp Fiction",
        "year": 1994,
        "genres": ["Crime", "Drama"],
        "rating": 8.9,
        "poster": "https://image.tmdb.org/t/p/w500/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/suaEOtk1N1sgg2MTM7oZd2cfVp3.jpg",
        "description": "The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption.",
        "duration": 154,
        "director": "Quentin Tarantino",
        "cast": ["John Travolta", "Uma Thurman", "Samuel L. Jackson"],
        "popularity": 91.7,
        "media_type": "movie"
    },
    {
        "id": 56,
        "title": "Gladiator",
        "year": 2000,
        "genres": ["Action", "Adventure", "Drama"],
        "rating": 8.5,
        "poster": "https://image.tmdb.org/t/p/w500/ty8TGRuvJLPUmAR1H1nRIsgwvim.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/xANuKR7xcMjEqM0TVXSZRmw5WKd.jpg",
        "description": "A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family.",
        "duration": 155,
        "director": "Ridley Scott",
        "cast": ["Russell Crowe", "Joaquin Phoenix", "Connie Nielsen"],
        "popularity": 89.8,
        "media_type": "movie"
    },
    {
        "id": 57,
        "title": "The Departed",
        "year": 2006,
        "genres": ["Crime", "Drama", "Thriller"],
        "rating": 8.5,
        "poster": "https://image.tmdb.org/t/p/w500/nT97ifVT2J1yMQmeq20Qblg61T.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/5Fh8CUI2IFmZyGXcjbwAoZN3TcM.jpg",
        "description": "An undercover cop and a mole in the police attempt to identify each other while infiltrating an Irish gang.",
        "duration": 151,
        "director": "Martin Scorsese",
        "cast": ["Leonardo DiCaprio", "Matt Damon", "Jack Nicholson"],
        "popularity": 87.4,
        "media_type": "movie"
    },
    {
        "id": 58,
        "title": "The Prestige",
        "year": 2006,
        "genres": ["Drama", "Mystery", "Sci-Fi"],
        "rating": 8.5,
        "poster": "https://image.tmdb.org/t/p/w500/tRNlZbgNCNOpLpbPEz5L8G8A0JN.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/mWB1nQsKk2ReYbzVfmME0yqILaX.jpg",
        "description": "After a tragic accident, two stage magicians engage in a battle to create the ultimate illusion.",
        "duration": 130,
        "director": "Christopher Nolan",
        "cast": ["Christian Bale", "Hugh Jackman", "Scarlett Johansson"],
        "popularity": 86.9,
        "media_type": "movie"
    },
    {
        "id": 59,
        "title": "Django Unchained",
        "year": 2012,
        "genres": ["Drama", "Western"],
        "rating": 8.4,
        "poster": "https://image.tmdb.org/t/p/w500/7oWY8VDWW7thTzWh3OKYRkWUlD5.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/2oZklIzUbvZXXzIFzv7Hi68d6xf.jpg",
        "description": "With the help of a German bounty-hunter, a freed slave sets out to rescue his wife from a brutal plantation owner.",
        "duration": 165,
        "director": "Quentin Tarantino",
        "cast": ["Jamie Foxx", "Christoph Waltz", "Leonardo DiCaprio"],
        "popularity": 88.6,
        "media_type": "movie"
    },
    {
        "id": 60,
        "title": "Inglourious Basterds",
        "year": 2009,
        "genres": ["Adventure", "Drama", "War"],
        "rating": 8.3,
        "poster": "https://image.tmdb.org/t/p/w500/7sfbEnaARXDDhKm0CZ7D7uc2sbo.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/gLbBRyS7MBrmVUNce91Hmx9vzqI.jpg",
        "description": "In Nazi-occupied France, a group of Jewish-American soldiers known as 'The Basterds' plan a cinema massacre.",
        "duration": 153,
        "director": "Quentin Tarantino",
        "cast": ["Brad Pitt", "Christoph Waltz", "Mélanie Laurent"],
        "popularity": 87.2,
        "media_type": "movie"
    },
]

# Add more movies dynamically
def generate_additional_movies():
    """Generate additional movies to reach 100+"""
    additional = []
    base_id = 61
    
    extra_movies = [
        ("No Country for Old Men", 2007, ["Crime", "Drama", "Thriller"], 8.2, "Coen Brothers"),
        ("There Will Be Blood", 2007, ["Drama"], 8.2, "Paul Thomas Anderson"),
        ("The Green Mile", 1999, ["Crime", "Drama", "Fantasy"], 8.6, "Frank Darabont"),
        ("Goodfellas", 1990, ["Crime", "Drama"], 8.7, "Martin Scorsese"),
        ("Casino Royale", 2006, ["Action", "Adventure", "Thriller"], 8.0, "Martin Campbell"),
        ("The Revenant", 2015, ["Adventure", "Drama", "Thriller"], 8.0, "Alejandro González Iñárritu"),
        ("The Departed", 2006, ["Crime", "Drama", "Thriller"], 8.5, "Martin Scorsese"),
        ("Léon: The Professional", 1994, ["Action", "Crime", "Drama"], 8.5, "Luc Besson"),
        ("American Beauty", 1999, ["Drama"], 8.3, "Sam Mendes"),
        ("Memento", 2000, ["Mystery", "Thriller"], 8.4, "Christopher Nolan"),
        ("Saving Private Ryan", 1998, ["Drama", "War"], 8.6, "Steven Spielberg"),
        ("City of God", 2002, ["Crime", "Drama"], 8.6, "Fernando Meirelles"),
        ("The Pianist", 2002, ["Drama", "War"], 8.5, "Roman Polanski"),
        ("The Usual Suspects", 1995, ["Crime", "Mystery", "Thriller"], 8.5, "Bryan Singer"),
        ("American History X", 1998, ["Crime", "Drama"], 8.5, "Tony Kaye"),
        ("Oldboy", 2003, ["Action", "Drama", "Mystery"], 8.4, "Park Chan-wook"),
        ("Once Upon a Time in Hollywood", 2019, ["Comedy", "Drama"], 7.6, "Quentin Tarantino"),
        ("The Truman Show", 1998, ["Comedy", "Drama"], 8.2, "Peter Weir"),
        ("A Beautiful Mind", 2001, ["Drama"], 8.2, "Ron Howard"),
        ("The Sixth Sense", 1999, ["Drama", "Mystery", "Thriller"], 8.1, "M. Night Shyamalan"),
        ("Heat", 1995, ["Action", "Crime", "Drama"], 8.3, "Michael Mann"),
        ("V for Vendetta", 2005, ["Action", "Drama", "Sci-Fi"], 8.2, "James McTeigue"),
        ("Gran Torino", 2008, ["Drama"], 8.1, "Clint Eastwood"),
        ("Catch Me If You Can", 2002, ["Biography", "Crime", "Drama"], 8.1, "Steven Spielberg"),
        ("12 Years a Slave", 2013, ["Biography", "Drama", "History"], 8.1, "Steve McQueen"),
        ("The Big Short", 2015, ["Biography", "Comedy", "Drama"], 7.8, "Adam McKay"),
        ("Arrival", 2016, ["Drama", "Mystery", "Sci-Fi"], 7.9, "Denis Villeneuve"),
        ("Her", 2013, ["Drama", "Romance", "Sci-Fi"], 8.0, "Spike Jonze"),
        ("Ex Machina", 2014, ["Drama", "Sci-Fi", "Thriller"], 7.7, "Alex Garland"),
        ("The Shape of Water", 2017, ["Adventure", "Drama", "Fantasy"], 7.3, "Guillermo del Toro"),
        ("Three Billboards Outside Ebbing, Missouri", 2017, ["Crime", "Drama"], 8.1, "Martin McDonagh"),
        ("Marriage Story", 2019, ["Comedy", "Drama", "Romance"], 7.9, "Noah Baumbach"),
        ("The Irishman", 2019, ["Biography", "Crime", "Drama"], 7.8, "Martin Scorsese"),
        ("Soul", 2020, ["Animation", "Adventure", "Comedy"], 8.0, "Pete Docter"),
        ("Nomadland", 2020, ["Drama"], 7.3, "Chloé Zhao"),
        ("Everything Everywhere All at Once", 2022, ["Action", "Adventure", "Comedy"], 7.8, "Daniel Kwan"),
        ("Oppenheimer", 2023, ["Biography", "Drama", "History"], 8.4, "Christopher Nolan"),
        ("Barbie", 2023, ["Adventure", "Comedy", "Fantasy"], 7.0, "Greta Gerwig"),
        ("Poor Things", 2023, ["Comedy", "Drama", "Romance"], 7.9, "Yorgos Lanthimos"),
        ("Killers of the Flower Moon", 2023, ["Crime", "Drama", "History"], 7.7, "Martin Scorsese"),
    ]
    
    # Placeholder posters for additional movies
    placeholder_posters = [
        "https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=300&h=450&fit=crop",
        "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=300&h=450&fit=crop",
        "https://images.unsplash.com/photo-1440404653325-ab127d49abc1?w=300&h=450&fit=crop",
        "https://images.unsplash.com/photo-1485846234645-a62644f84728?w=300&h=450&fit=crop",
        "https://images.unsplash.com/photo-1478720568477-152d9b164e26?w=300&h=450&fit=crop",
    ]
    
    for i, (title, year, genres, rating, director) in enumerate(extra_movies):
        additional.append({
            "id": base_id + i,
            "title": title,
            "year": year,
            "genres": genres,
            "rating": rating,
            "poster": placeholder_posters[i % len(placeholder_posters)],
            "backdrop": placeholder_posters[i % len(placeholder_posters)],
            "description": f"A critically acclaimed {genres[0].lower()} film directed by {director}.",
            "duration": 120 + (i % 60),
            "director": director,
            "cast": ["Actor 1", "Actor 2", "Actor 3"],
            "popularity": 75 + random.random() * 20,
            "media_type": "movie"
        })
    
    return additional


# Complete database
MEDIA_DATABASE.extend(generate_additional_movies())


def get_all_media() -> List[Dict[str, Any]]:
    """Get all media items."""
    return MEDIA_DATABASE


def get_media_by_id(media_id: int) -> Optional[Dict[str, Any]]:
    """Get a specific media item by ID."""
    for item in MEDIA_DATABASE:
        if item["id"] == media_id:
            return item
    return None


def get_media_by_genre(genre: str) -> List[Dict[str, Any]]:
    """Get media items by genre."""
    return [item for item in MEDIA_DATABASE if genre in item["genres"]]


def get_recommendations_for_user(
    user_id: int,
    n: int = 10,
    genres: Optional[List[str]] = None,
    exclude_ids: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """
    Generate personalized recommendations for a user.
    
    This simulates what the ML model would return.
    """
    # Use user_id as seed for consistent results
    random.seed(user_id)
    
    # Filter by genres if specified
    candidates = MEDIA_DATABASE.copy()
    if genres:
        candidates = [m for m in candidates if any(g in m["genres"] for g in genres)]
    
    # Exclude specified IDs
    if exclude_ids:
        candidates = [m for m in candidates if m["id"] not in exclude_ids]
    
    # Shuffle and select
    random.shuffle(candidates)
    selected = candidates[:n]
    
    # Add recommendation scores
    for i, item in enumerate(selected):
        item = item.copy()
        item["score"] = 0.95 - (i * 0.03) + random.random() * 0.05
        item["reason"] = random.choice([
            f"Because you enjoyed {genres[0] if genres else 'similar'} content",
            "Highly rated by users with similar taste",
            "Trending in your region",
            "Matches your viewing pattern",
            "Popular among your age group",
            "Award-winning content you might like"
        ])
        selected[i] = item
    
    return selected


def get_similar_items(item_id: int, n: int = 10) -> List[Dict[str, Any]]:
    """Get items similar to a given item."""
    source = get_media_by_id(item_id)
    if not source:
        return []
    
    # Find items with matching genres
    similar = []
    for item in MEDIA_DATABASE:
        if item["id"] == item_id:
            continue
        
        # Calculate similarity based on genre overlap
        common_genres = set(source["genres"]) & set(item["genres"])
        if common_genres:
            item_copy = item.copy()
            item_copy["similarity"] = len(common_genres) / len(set(source["genres"]) | set(item["genres"]))
            similar.append(item_copy)
    
    # Sort by similarity and rating
    similar.sort(key=lambda x: (x["similarity"], x["rating"]), reverse=True)
    
    return similar[:n]


def get_trending(n: int = 10) -> List[Dict[str, Any]]:
    """Get trending items."""
    sorted_items = sorted(MEDIA_DATABASE, key=lambda x: x["popularity"], reverse=True)
    return sorted_items[:n]


def get_top_rated(n: int = 10) -> List[Dict[str, Any]]:
    """Get top rated items."""
    sorted_items = sorted(MEDIA_DATABASE, key=lambda x: x["rating"], reverse=True)
    return sorted_items[:n]


# Export
__all__ = [
    "MEDIA_DATABASE",
    "get_all_media",
    "get_media_by_id",
    "get_media_by_genre",
    "get_recommendations_for_user",
    "get_similar_items",
    "get_trending",
    "get_top_rated"
]
