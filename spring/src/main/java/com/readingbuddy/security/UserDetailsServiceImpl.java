package com.readingbuddy.security;

import com.readingbuddy.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

// Spring Security requires a UserDetailsService to load user credentials for authentication.
// We store the user's UUID as the principal "username" so that auth.getName() returns the ID,
// and any service can resolve the full entity with userRepository.findById(auth.getName()).
@Service
public class UserDetailsServiceImpl implements UserDetailsService {

    @Autowired
    private UserRepository userRepository;

    // Called by the standard form-login path (not used here, but required by the interface)
    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        return userRepository.findByUsername(username)
                .map(this::toUserDetails)
                .orElseThrow(() -> new UsernameNotFoundException("User not found: " + username));
    }

    // Called by JwtCookieFilter after extracting the user ID from the JWT
    public UserDetails loadUserById(String id) {
        return userRepository.findById(id)
                .map(this::toUserDetails)
                .orElseThrow(() -> new UsernameNotFoundException("User not found: " + id));
    }

    private UserDetails toUserDetails(com.readingbuddy.entity.User user) {
        return User.builder()
                .username(user.getId())           // ID as principal name
                .password(user.getHashedPassword())
                .disabled(!user.isActive())
                .roles("USER")
                .build();
    }
}
