package com.readingbuddy.security;

import com.readingbuddy.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

@Service
public class UserDetailsServiceImpl implements UserDetailsService {

    @Autowired
    private UserRepository userRepository;

    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        return userRepository.findByUsername(username)
                .map(this::toUserDetails)
                .orElseThrow(() -> new UsernameNotFoundException("User not found: " + username));
    }

    public UserDetails loadUserById(String id) {
        return userRepository.findById(id)
                .map(this::toUserDetails)
                .orElseThrow(() -> new UsernameNotFoundException("User not found: " + id));
    }

    private UserDetails toUserDetails(com.readingbuddy.entity.User user) {
        return User.builder()
                .username(user.getId())
                .password(user.getHashedPassword())
                .disabled(!user.isActive())
                .roles("USER")
                .build();
    }
}
