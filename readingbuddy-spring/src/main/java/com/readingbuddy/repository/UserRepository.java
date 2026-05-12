package com.readingbuddy.repository;

import com.readingbuddy.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

// Python equivalent: get_user_by_id / get_user_by_username / get_user_by_email in models.py
// Spring Data JPA generates the SQL from the method name — no manual query needed.
public interface UserRepository extends JpaRepository<User, String> {

    Optional<User> findByUsername(String username);

    Optional<User> findByEmail(String email);
}
